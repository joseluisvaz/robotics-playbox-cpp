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

#include "math.hpp"
#include "types.hpp"
#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <thread>

namespace RoboticsSandbox
{
using namespace Eigen;

/* Implementation of a simple Cross Entropy Method Model Predictive Control (CEM-MPC) */
class IntelligentDriverModel
{

public:
  using DynamicsT = SingleIntegrator;
  using State = typename DynamicsT::State;
  using Action = typename DynamicsT::Action;
  using States = typename DynamicsT::States;
  using Actions = typename DynamicsT::Actions;
  using Trajectory = typename DynamicsT::Trajectory;
  using Sampler = NormalRandomVariable<Actions>;
  constexpr static int state_size = DynamicsT::state_size;
  constexpr static int action_size = DynamicsT::action_size;

  struct Config
  {
    float a = 1.5f;   // max acceleration
    float la = 4.0f;  // length of vehicle_in_front
    float v0 = 10.0f; // desired speed
    float s0 = 2.0f;  // min spacing
    float T = 2.0f;   // desired time gap
    float b = 2.0f;   // comfortable braking decceleration (positive)
    int N = 10;       // horizon
    int delta = 4;
  };

  IntelligentDriverModel() = default;

  // Instantiate an intelligent driver model.
  IntelligentDriverModel(const Config &config) : config_(std::move(config)){};

  // rollout a trajectory for this model.
  States& rollout(const Ref<const State> &x0, const Ref<const States> &x_lead_states)
  {
    states_ = States::Zero(state_size, x_lead_states.cols());
    states_.col(0) = x0;
    for (int i = 0; i + 1 < config_.N; ++i)
    {
      Action action = get_action(states_.col(i), x_lead_states.col(i));
      SingleIntegrator::step(states_.col(i), action, states_.col(i + 1));
    }

    return states_;
  };

  States& get_states()
  {
    return states_;
  };

  // get action from state (controller/policy).
  Action get_action(const Ref<const State> &x, const Ref<const State> &x_lead)
  {
    Action action = Action::Zero();
    const float sa = x_lead[0] - x[0] - config_.la;
    const float delta_va = x[1] - x_lead[1];
    const float s_star = config_.s0 + x[1] * config_.T + x[1] * delta_va / (2 * std::sqrt(config_.a * config_.b));
    action[0] = config_.a * (1 - std::pow(x[1] / config_.v0, 4) - std::pow(s_star / sa, 2));
    return action;
  };

  SingleIntegrator model_;
  Config config_;
  States states_;
};

} // namespace RoboticsSandbox
