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

#include <Eigen/Dense>
#include <algorithm>

#include "math.hpp"

namespace GuiApplication
{

using namespace Eigen;

template <int state_size>
using EigenState = Vector<float, state_size>;

template <int action_size>
using EigenAction = Vector<float, action_size>;

template <int state_size>
using EigenStateSequence = Matrix<float, state_size, Dynamic>;

template <int action_size>
using EigenActionSequence = Matrix<float, action_size, Dynamic>;

template <int state_size, int action_size>
struct EigenTrajectory
{
  std::vector<float> times{};
  EigenStateSequence<state_size> states;
  EigenActionSequence<action_size> actions;
};

template <int _state_size, int _action_size>
class Dynamics
{
public:
  constexpr static int state_size = _state_size;
  constexpr static int action_size = _action_size;

  using State = EigenState<state_size>;
  using Action = EigenAction<action_size>;
  using States = EigenStateSequence<state_size>;
  using Actions = EigenActionSequence<action_size>;
  using Trajectory = EigenTrajectory<state_size, action_size>;
};

class EigenKinematicBicycle : public Dynamics<6, 2>
{

public:
  constexpr static float ts{0.5f};
  constexpr static float one_over_wheelbase{1.0f / 3.0f};
  constexpr static float max_steering{0.52f};
  constexpr static float max_steering_rate{0.1f};
  constexpr static float max_jerk{0.6f};

  static void step(const Ref<const State> &state, const Ref<const Action> &action, Ref<State> new_state)
  {
    new_state[0] = state[0] + ts * state[3] * std::cos(state[2]);
    new_state[1] = state[1] + ts * state[3] * std::sin(state[2]);
    new_state[2] = state[2] + ts * state[3] * one_over_wheelbase * std::tan(state[5]);
    new_state[3] = state[3] + ts * state[4];
    new_state[4] = state[4] + ts * max_jerk * std::tanh(action[0]);
    new_state[5] = std::clamp(state[5] + ts * max_steering_rate * std::tanh(action[1]), -max_steering, max_steering);
  }
};

struct CostFunction
{
  using D = EigenKinematicBicycle;

  float evaluate_state_action_pair(const Ref<const D::State> &state, const Ref<const D::Action> &action) const
  {

    return state_slider_values_[0] * (state[0] - ref_values_[0]) * (state[0] - ref_values_[0]) +
           state_slider_values_[1] * (state[1] - ref_values_[1]) * (state[1] - ref_values_[1]) +
           state_slider_values_[2] * (ANGLE_DIFF(state[2], ref_yaw_)) * (ANGLE_DIFF(state[2], ref_yaw_)) +
           state_slider_values_2_[0] * (state[3] - ref_values_[2]) * (state[3] - ref_values_[2]) +
           state_slider_values_2_[1] * (state[4] * state[4]) + state_slider_values_2_[2] * (state[5] - state[5]) +
           action_slider_values_[0] * (action[0] - action[0]) + action_slider_values_[1] * (action[1] - action[1]);
  }

  float evaluate_terminal_pair(const Ref<const D::State> &state, const Ref<const D::Action> &action) const
  {

    return terminal_state_slider_values_[0] * (state[0] - terminal_ref_values_[0]) *
               (state[0] - terminal_ref_values_[0]) +
           terminal_state_slider_values_[1] * (state[1] - terminal_ref_values_[1]) *
               (state[1] - terminal_ref_values_[1]) +
           terminal_state_slider_values_[2] * (ANGLE_DIFF(state[2], terminal_ref_yaw_)) *
               (ANGLE_DIFF(state[2], terminal_ref_yaw_)) +
           terminal_state_slider_values_2_[0] * (state[3] - terminal_ref_values_[2]) *
               (state[3] - terminal_ref_values_[2]) +
           terminal_state_slider_values_2_[1] * (state[4] * state[4]) +
           terminal_state_slider_values_2_[2] * (state[5] - state[5]) +
           terminal_action_slider_values_[0] * (action[0] - action[0]) +
           terminal_action_slider_values_[1] * (action[1] - action[1]);
  }

  float operator()(const Ref<const D::States> &states, const Ref<const D::Actions> &actions) const
  {
    float cost = 0.0f;
    for (int i{0}; i + 1 < states.cols(); ++i)
    {
      cost += evaluate_state_action_pair(states.col(i), actions.col(i));
    }
    cost += evaluate_terminal_pair(states.col(states.cols() - 1), actions.col(actions.cols() - 1));
    return cost;
  }

  float state_slider_values_[3]{1.0f, 1.0f, 1.0f};
  float state_slider_values_2_[3]{1.0f, 1.0f, 1.0f};
  float action_slider_values_[2]{1.0f, 1.0f};
  float ref_values_[3]{0.0f, 0.0f, 0.0f};
  float ref_yaw_{0.0f};

  float terminal_state_slider_values_[3]{1.0f, 1.0f, 1.0f};
  float terminal_state_slider_values_2_[3]{1.0f, 1.0f, 1.0f};
  float terminal_action_slider_values_[2]{1.0f, 1.0f};
  float terminal_ref_values_[3]{0.0f, 0.0f, 0.0f};
  float terminal_ref_yaw_{0.0f};
};

} // namespace GuiApplication