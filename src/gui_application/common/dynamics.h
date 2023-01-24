// Copyright © 2022 <copyright holders>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the “Software”), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE

#pragma once

#include <cuda_runtime.h>
// #ifndef __host__
// #define __host__
// #endif
//
// #ifndef __device__
// #define __device__
// #endif

#include "cmath"
#include "types.hpp"

namespace mpex
{

template <int _state_size, int _action_size, typename T = double>
class Dynamics
{
public:
  constexpr static int state_size = _state_size;
  constexpr static int action_size = _action_size;
  
  using FloatT = T;

  using State = EigenState<state_size, T>;
  using Action = EigenAction<action_size, T>;
  using States = EigenStateSequence<state_size, T>;
  using Actions = EigenActionSequence<action_size, T>;
  using Trajectory = EigenTrajectory<state_size, action_size>;
};

template <int _state_size, int _action_size>
using DynamicsF = Dynamics<_state_size, _action_size, float>;
template <int _state_size, int _action_size>
using DynamicsD = Dynamics<_state_size, _action_size, double>;

template <typename T>
class EigenKinematicBicycleT : public Dynamics<6, 2, T>
{

  using D = Dynamics<6, 2, T>;

public:
  constexpr static T ts{0.5f};
  constexpr static T one_over_wheelbase{1.0f / 3.0f};
  constexpr static T max_steering{0.52f};
  constexpr static T max_steering_rate{0.1f};
  constexpr static T max_jerk{1.0f};

  template <typename VectorT>
  static VectorT step_diff(const VectorT &state, const VectorT &action)
  {
    VectorT new_state = VectorT::Zero(state.size());
    new_state[0] = state[0] + ts * state[3] * cos(state[2]);
    new_state[1] = state[1] + ts * state[3] * sin(state[2]);
    new_state[2] = state[2] + ts * state[3] * one_over_wheelbase * tan(state[5]);
    new_state[3] = state[3] + ts * state[4];
    new_state[4] = state[4] + ts * max_jerk * tanh(action[0]);
    new_state[5] = state[5] + ts * max_steering_rate * tanh(action[1]);
    return new_state;
  }

  __host__ __device__ static void cu_step(
      const T *state,
      const T *action,
      double *new_state,
      const T ts,
      const T one_over_wheelbase,
      const T max_jerk,
      const T max_steering_rate)
  {

    new_state[0] = state[0] + ts * state[3] * cos(state[2]);
    new_state[1] = state[1] + ts * state[3] * sin(state[2]);
    new_state[2] = state[2] + ts * state[3] * one_over_wheelbase * tan(state[5]);
    new_state[3] = state[3] + ts * state[4];
    new_state[4] = state[4] + ts * max_jerk * tanh(action[0]);
    new_state[5] = state[5] + ts * max_steering_rate * tanh(action[1]);
  };

  static void step(const Ref<const typename D::State> &state, const Ref<const typename D::Action> &action, Ref<typename D::State> new_state)
  {
    cu_step(state.data(), action.data(), new_state.data(), ts, one_over_wheelbase, max_jerk, max_steering_rate);
  }

  static typename D::State step_(const Ref<const typename D::State> &state, const Ref<const typename D::Action> &action)
  {
    typename D::State new_state = D::State::Zero();
    step(state, action, new_state);
    return new_state;
  }
};

using EigenKinematicBicycle = EigenKinematicBicycleT<double>;
using EigenKinematicBicycleF = EigenKinematicBicycleT<float>;

template <typename T>
class SingleIntegratorT : public Dynamics<2, 1, T>
{

  using D = Dynamics<2, 1, T>;

public:
  constexpr static T ts{0.5f};
  __host__ __device__ static void step_single(const T &x, const T &v, const T &a, T &next_x, T &next_v)
  {
    next_x = x + ts * v + (ts * ts) / 2.0 * a;
    next_x = x + ts * v;
    next_v = v + ts * a;
  }
  __host__ __device__ static void cu_step(const T *state, const T *action, T *new_state)
  {
    step_single(state[0], state[1], action[0], new_state[0], new_state[1]);
  }

  static void step(const Ref<const typename D::State> &state, const Ref<const typename D::Action> &action, Ref<typename D::State> new_state)
  {
    cu_step(state.data(), action.data(), new_state.data());
  }

  static typename D::State step_(const Ref<const typename D::State> &state, const Ref<const typename D::Action> &action)
  {
    typename D::State new_state = D::State::Zero();
    step(state, action, new_state);
    return new_state;
  }
};

using SingleIntegrator = SingleIntegratorT<double>;
using SingleIntegratorF = SingleIntegratorT<float>;

// class SingleIntegratorF : public Dynamics<2, 1, float>
// {
// public:
//   constexpr static double ts{0.5f};
//   __host__ __device__ static void step_single(const float &x, const float &v, const float &a, float &next_x, float &next_v)
//   {
//     constexpr float ts = 0.1f;
//     next_x = x + ts * v + (ts * ts) / 2.0f * a;
//     next_x = x + ts * v;
//     next_v = v + ts * a;
//   }
//   __host__ __device__ static void cu_step(const float *state, const float *action, float *new_state)
//   {
//     step_single(state[0], state[1], action[0], new_state[0], new_state[1]);
//   }
//
//   static void step(const Ref<const State> &state, const Ref<const Action> &action, Ref<State> new_state)
//   {
//     cu_step(state.data(), action.data(), new_state.data());
//   }
//
//   static State step_(const Ref<const State> &state, const Ref<const Action> &action)
//   {
//     State new_state = State::Zero();
//     step(state, action, new_state);
//     return new_state;
//   }
// };

} // namespace mpex