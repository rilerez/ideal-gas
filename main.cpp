#include <vector>
#include <complex>
#include <random>
#include <iostream>
#include <cassert>
#include <cmath>

#include "sdl2raii/emscripten_glue.hpp"
#include "sdl2raii/sdl.hpp"

template<class Thunk>
struct finally {
  Thunk thunk;
  finally(Thunk thunk) : thunk{std::move(thunk)} {}
  ~finally() { thunk(); }
};

using namespace std::literals;
namespace chrono = std::chrono;

using fptype = double;
using vec = std::complex<fptype>;

auto dot(vec const z, vec const w) { return z * conj(w); }
auto wedge(vec const z, vec const w) {
  return z.real() * w.imag() - z.imag() * w.real();
}

std::vector<vec> position;
std::vector<vec> velocity;

fptype radius = 5;

int world_width = 300;
int world_height = world_width;

auto constexpr update_step = 20ms;

inline auto clamp(fptype low, fptype high, fptype x) {
  return std::max(low, std::min(high, x));
}
#define FN(...) [&](auto _) { return __VA_ARGS__; }
vec in_bounds(vec x) {
  auto clamper = FN(clamp(radius, world_height - radius, _));
  return {clamper(x.real()), clamper(x.imag())};
}

const fptype col_rad = 5 * radius;
bool is_collide(int i1, int i2) {
  return norm(position[i1] - position[i2]) <= col_rad;
}

void collide_update(int i1, int i2) {
  auto const v1 = velocity[i1];
  auto const v2 = velocity[i2];
  auto const p1 = position[i1];
  auto const p2 = position[i2];
  // prevent division by 0
  constexpr fptype smooth = .0001;
  constexpr fptype offset = .0005;
  constexpr auto collide1 =
      [=](vec const v1, vec const v2, vec const p1, vec const p2) {
        auto const d = (p1 - p2);
        auto const u = (d + offset) / (norm(d) + smooth);
        return make_tuple(v1 - (dot(v1 - v2, u) * d), p1 + u * col_rad * .7);
      };
  std::tie(velocity[i1], position[i1]) = collide1(v1, v2, p1, p2);
  std::tie(velocity[i2], position[i2]) = collide1(v2, v1, p2, p1);
}

void keep_in_bounds(int i) {
  if(position[i].real() <= radius || position[i].real() >= world_width - radius)
    velocity[i] = vec{-velocity[i].real(), velocity[i].imag()};
  if(position[i].imag() <= radius
     || position[i].imag() >= world_height - radius)
    velocity[i] = vec{velocity[i].real(), -velocity[i].imag()};
  position[i] = in_bounds(position[i]);
}

void update() {
  for(int i = 0; i < position.size(); ++i) {
    position[i] += velocity[i] * static_cast<fptype>(update_step.count());
    keep_in_bounds(i);
  }
  for(int i = 0; i < position.size(); ++i)
    for(int j = 0; j < i; ++j)
      if(is_collide(i, j))
        collide_update(i, j);
}

sdl::unique::Texture tex;
void render(sdl::Renderer* renderer, chrono::milliseconds lag) {
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

  constexpr auto max_speed = .03;
  auto rand_vel = std::uniform_real_distribution<fptype>(-max_speed, max_speed);

  constexpr int num_things = 400;
  position.resize(num_things);
  velocity.resize(num_things);

  for(int i = 0; i < num_things; ++i)
    position[i] = {rand_pos(*gen), rand_pos(*gen)};
  for(int i = 0; i < num_things; ++i)
    velocity[i] = {rand_vel(*gen), rand_vel(*gen)};

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

  auto last_time = chrono::high_resolution_clock::now();
  auto lag = last_time - last_time;

  emscripten_glue::main_loop([&] {
    auto this_time = chrono::high_resolution_clock::now();
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

    render(renderer.get(), chrono::duration_cast<chrono::milliseconds>(lag));

    last_time = this_time;
  });
  return 0;
}
