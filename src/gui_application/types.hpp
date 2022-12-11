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
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include "math.hpp"
#include <iostream>

namespace RoboticsSandbox
{

using namespace Eigen;
using namespace autodiff;

/*
  Defining common linear algebra types that we are going to use in this project
*/
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

template <typename EigenT>
using VectorOfEigen = std::vector<EigenT, Eigen::aligned_allocator<EigenT>>;

template <int state_size, typename T = double>
using EigenState = Eigen::Vector<T, state_size>;

template <int action_size, typename T = double>
using EigenAction = Eigen::Vector<T, action_size>;

template <int state_size, typename T = double>
using EigenStateSequence = Eigen::Matrix<T, state_size, Eigen::Dynamic>;

template <int action_size, typename T = double>
using EigenActionSequence = Eigen::Matrix<T, action_size, Eigen::Dynamic>;

template <int state_size, int action_size, typename T = double>
struct EigenTrajectory
{
  std::vector<T> times{};
  EigenStateSequence<state_size, T> states;
  EigenActionSequence<action_size, T> actions;

  EigenTrajectory() = default;
  EigenTrajectory(int horizon)
  {
    states = EigenStateSequence<state_size, T>::Zero(state_size, horizon);
    actions = EigenActionSequence<action_size, T>::Zero(action_size, horizon);
    times = std::vector<T>(horizon, 0.0f);
  };

  EigenState<state_size, T> state_at(int i) const
  {
    return this->states.col(i);
  };

  EigenState<action_size, T> action_at(int i) const
  {
    return this->actions.col(i);
  };
};

template <int _state_size, int _action_size, typename T = double>
class Dynamics
{
public:
  constexpr static int state_size = _state_size;
  constexpr static int action_size = _action_size;

  using State = EigenState<state_size, T>;
  using Action = EigenAction<action_size, T>;
  using States = EigenStateSequence<state_size, T>;
  using Actions = EigenActionSequence<action_size, T>;
  using Trajectory = EigenTrajectory<state_size, action_size>;
};

class EigenKinematicBicycle : public Dynamics<6, 2, double>
{

public:
  constexpr static double ts{0.5};
  constexpr static double one_over_wheelbase{1.0 / 3.0};
  constexpr static double max_steering{0.52};
  constexpr static double max_steering_rate{0.1};
  constexpr static double max_jerk{1.0};

  static State step_(const Ref<const State> &state, const Ref<const Action> &action)
  {
    State new_state = State::Zero();
    new_state[0] = state[0] + ts * state[3] * std::cos(state[2]);
    new_state[1] = state[1] + ts * state[3] * std::sin(state[2]);
    new_state[2] = state[2] + ts * state[3] * one_over_wheelbase * std::tan(state[5]);
    new_state[3] = state[3] + ts * state[4];
    new_state[4] = state[4] + ts * max_jerk * std::tanh(action[0]);
    new_state[5] = state[5] + ts * max_steering_rate * std::tanh(action[1]);
    return new_state;
  }

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

  // static void step(const Ref<const State> &state, const Ref<const Action> &action, Ref<State> new_state)
  // {
  //   new_state[0] = state[0] + ts * state[3] * std::cos(state[2]);
  //   new_state[1] = state[1] + ts * state[3] * std::sin(state[2]);
  //   new_state[2] = state[2] + ts * state[3] * one_over_wheelbase * std::tan(state[5]);
  //   new_state[3] = state[3] + ts * state[4];
  //   new_state[4] = state[4] + ts * max_jerk * std::tanh(action[0]);
  //   new_state[5] = std::clamp(state[5] + ts * max_steering_rate * std::tanh(action[1]), -max_steering, max_steering);
  // }
  static void step(const Ref<const State> &state, const Ref<const Action> &action, Ref<State> new_state)
  {
    new_state[0] = state[0] + ts * state[3] * std::cos(state[2]);
    new_state[1] = state[1] + ts * state[3] * std::sin(state[2]);
    new_state[2] = state[2] + ts * state[3] * one_over_wheelbase * std::tan(state[5]);
    new_state[3] = state[3] + ts * state[4];
    new_state[4] = state[4] + ts * max_jerk * std::tanh(action[0]);
    new_state[5] = state[5] + ts * max_steering_rate * std::tanh(action[1]);
  }
};

class SingleIntegrator : public Dynamics<2, 1>
{
public:
  constexpr static double ts{0.5f};
  static void step(const Ref<const State> &state, const Ref<const Action> &action, Ref<State> new_state)
  {
    new_state[0] = state[0] + ts * state[1];
    new_state[1] = state[1] + ts * action[0];
  }
};

struct CostFunction
{
  using D = EigenKinematicBicycle;

  double evaluate_state_action_pair(const Ref<const D::State> &s, const Ref<const D::Action> &a, const double w_s[],
                                    const double w_a[], const double r_s[], const double r_a[]) const
  {
    double cost = 0.0f;
    for (int i = 0; i < D::state_size; ++i)
    {
      if (i == 2 || i == 5)
      {
        // It is the yaw state or steering
        cost += w_s[2] * (ANGLE_DIFF(s[2], r_s[2])) * (ANGLE_DIFF(s[2], r_s[2]));
        continue;
      }
      cost += w_s[i] * (s[i] - r_s[i]) * (s[i] - r_s[i]);
    }

    for (int i = 0; i < D::action_size; ++i)
    {
      cost += w_a[i] * (a[i] - r_a[i]) * (a[i] - r_a[i]);
    }

    return cost;
  }

  double operator()(const Ref<const D::States> &states, const Ref<const D::Actions> &actions) const
  {
    double cost = 0.0f;
    for (int i{0}; i + 1 < states.cols(); ++i)
    {
      cost += evaluate_state_action_pair(states.col(i), actions.col(i), w_s_, w_a_, r_s_, r_a_);
    }
    cost += evaluate_state_action_pair(states.col(states.cols() - 1), actions.col(actions.cols() - 1UL), W_s_, W_a_,
                                       R_s_, R_a_);
    return cost;
  }

  double w_s_[D::state_size]{0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // state costs
  double w_a_[D::action_size]{1.0f, 1.0f};                        // action costs

  double W_s_[D::state_size]{0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // terminal state costs
  double W_a_[D::action_size]{1.0f, 1.0f};                        // terminal action costs

  double r_s_[D::state_size]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // state reference
  double r_a_[D::state_size]{0.0f, 0.0f};                         // action reference

  double R_s_[D::state_size]{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // terminal state reference
  double R_a_[D::state_size]{0.0f, 0.0f};                         // terminal action reference
};

} // namespace RoboticsSandbox