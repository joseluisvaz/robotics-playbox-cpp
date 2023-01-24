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

#include "common/dynamics.h"
#include "common/types.hpp"

namespace mpex
{

class CostFunction
{
  using D = EigenKinematicBicycle;

public:
  virtual double operator()(const Ref<const D::States> &states, const Ref<const D::Actions> &actions) const = 0;
};

class QuadraticCostFunction : public CostFunction
{
public:
  using D = EigenKinematicBicycle;

  double evaluate_state_action_pair(
      const Ref<const D::State> &s,
      const Ref<const D::Action> &a,
      const double w_s[],
      const double w_a[],
      const double r_s[],
      const double r_a[]) const
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
    cost += evaluate_state_action_pair(states.col(states.cols() - 1), actions.col(actions.cols() - 1UL), W_s_, W_a_, R_s_, R_a_);
    return cost;
  }

  double w_s_[D::state_size]{0.0f, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f}; // state costs
  double w_a_[D::action_size]{1.0f, 1.0f};                          // action costs

  double W_s_[D::state_size]{0.0f, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f}; // terminal state costs
  double W_a_[D::action_size]{1.0f, 1.0f};                          // terminal action costs

  double r_s_[D::state_size]{0.0f, 0.0f, 0.0f, 8.0f, 0.0f, 0.0f}; // state reference
  double r_a_[D::state_size]{0.0f, 0.0f};                         // action reference

  double R_s_[D::state_size]{0.0f, 0.0f, 0.0f, 8.0f, 0.0f, 0.0f}; // terminal state reference
  double R_a_[D::state_size]{0.0f, 0.0f};                         // terminal action reference
};

} // namespace mpex