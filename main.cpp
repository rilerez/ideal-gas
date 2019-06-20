#include <vector>
#include <complex>
#include <random>
#include <iostream>
#include <cassert>
#include <cmath>

#include "sdl2raii/emscripten_glue.hpp"
#include "sdl2raii/sdl.hpp"

using namespace std::literals;

using fptype = double;
using vec = std::complex<fptype>;
std::vector<vec> position;
std::vector<vec> velocity;

fptype radius = 5;
constexpr fptype damp = 1;

int world_width = 500;
int world_height = world_width;

template<class F>
struct finally {
  F f;
  finally(F f) : f{f} {}
  ~finally() { f(); }
};

auto constexpr update_step = 16ms;

inline auto clamp(fptype low, fptype high, fptype x) {
  return std::max(low, std::min(high, x));
}
#define FN(...) [&](auto _) { return __VA_ARGS__; }
vec in_bounds(vec x) {
  auto clamper = FN(clamp(radius, world_height - radius, _));
  return {clamper(x.real()), clamper(x.imag())};
}

auto dot(vec x, vec y) { return (x * conj(y)).real(); }

const fptype col_rad = 3 * radius;
bool is_collide(int i1, int i2) {
  return norm(position[i1] - position[i2]) <= col_rad;
}

constexpr auto square = [](auto x) { return x * x; };

void collide_update(int i1, int i2) {
  auto const v1 = velocity[i1];
  auto const v2 = velocity[i2];
  auto const p1 = position[i1];
  auto const p2 = position[i2];

  // prevent division by 0
  constexpr fptype smooth= .0001;

  constexpr auto collide_vel =
      [smooth](vec const v1, vec const v2, vec const p1, vec const p2) -> vec {
    auto d = (p1 - p2);
    auto u = d / (norm(d) + smooth);
    return v1 - (dot(v1 - v2, u) * d);
  };

  constexpr auto collide_pos = [smooth](vec const p1, vec const p2) -> vec {
    return p1 - ((p1 - p2) / (norm(p1 - p2) + smooth)) * col_rad;
  };
  velocity[i1] = collide_vel(v1, v2, p1, p2);
  velocity[i2] = collide_vel(v2, v1, p1, p2);
  position[i1] = collide_pos(p1, p2);
  position[i2] = collide_pos(p2, p1);
}

void update() {
  for(int i = 0; i < position.size(); ++i) {
    position[i] += velocity[i] * static_cast<fptype>(update_step.count());
    if(position[i].real() <= radius
       || position[i].real() >= world_width - radius)
      velocity[i] = vec{-velocity[i].real(), velocity[i].imag()} * damp;
    if(position[i].imag() <= radius
       || position[i].imag() >= world_height - radius)
      velocity[i] = vec{velocity[i].real(), -velocity[i].imag()} * damp;
    position[i] = in_bounds(position[i]);
  }

  for(int i = 0; i < position.size(); ++i)
    for(int j = 0; j < position.size(); ++j) {
      if(i == j)
        continue;
      if(is_collide(i, j))
        collide_update(i, j);
    }
}

sdl::unique::Texture tex;
void render(sdl::Renderer* renderer, std::chrono::milliseconds lag) {
  auto particle_at = [](auto pos) {
    return sdl::Rect{static_cast<int>(pos.real() - radius),
                     static_cast<int>(pos.imag() - radius),
                     static_cast<int>(2 * radius),
                     static_cast<int>(2 * radius)};
  };

  sdl::SetRenderDrawColor(renderer, {50, 50, 50, 255});
  sdl::RenderClear(renderer);
  sdl::SetRenderDrawColor(renderer, {200, 200, 200, 255});
  for(int i = 0; i < position.size(); ++i) {
    auto pos = position[i] + velocity[i] * static_cast<fptype>(lag.count());
    sdl::RenderCopy(renderer, tex.get(), std::nullopt, particle_at(pos));
  }
  sdl::RenderPresent(renderer);
}

int main() {
  sdl::Init(sdl::init::video);
  finally _ = [] { sdl::Quit(); };

  std::random_device rd;
  auto gen = std::make_unique<std::mt19937>(rd());
  auto rand_pos = std::uniform_real_distribution<fptype>{
      static_cast<fptype>(radius),
      static_cast<fptype>(world_width - radius)};

  constexpr auto max_speed = .1;
  auto rand_vel = std::uniform_real_distribution<fptype>(-max_speed, max_speed);

  constexpr int num_things = 600;
  position.resize(num_things);
  velocity.resize(num_things);

  for(int i = 0; i < num_things; ++i) {
    position[i] = {rand_pos(*gen), rand_pos(*gen)};
  }
  for(int i = 0; i < num_things; ++i) {
    velocity[i] = {rand_vel(*gen), rand_vel(*gen)};
  }

  auto window = sdl::CreateWindow("ideal gas",
                                  sdl::window::pos_undefined,
                                  sdl::window::pos_undefined,
                                  world_width,
                                  world_height,
                                  sdl::window::resizable);
  auto renderer = sdl::CreateRenderer(
      window.get(),
      -1,
      sdl::renderer::accelerated | sdl::renderer::presentvsync);

  tex = sdl::CreateTextureFromSurface(renderer.get(),
                                      sdl::LoadBMP("assets/circle.bmp"));

  auto last_time = std::chrono::high_resolution_clock::now();
  auto lag = last_time - last_time;

  emscripten_glue::main_loop([&] {
    auto this_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = this_time - last_time;
    lag += elapsed_time;

    for(; lag >= update_step; lag -= update_step)
      update();

    while(auto const event = sdl::NextEvent()) {
      switch(event->type) {
        case SDL_QUIT:
          emscripten_glue::cancel_main_loop();
          break;
      }
    }

    render(renderer.get(),
           std::chrono::duration_cast<std::chrono::milliseconds>(lag));

    last_time = this_time;
  });
  return 0;
}
