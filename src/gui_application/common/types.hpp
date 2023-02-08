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

#include <algorithm>
#include <iostream>

#include <Eigen/Dense>

#include "math.hpp"

namespace mpex {

using namespace Eigen;

/*
  Defining common linear algebra types that we are going to use in this project
*/
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

template <typename EigenT>
using VectorOfEigen = std::vector<EigenT, Eigen::aligned_allocator<EigenT>>;

template <int state_size, typename T = double>
using EigenState = Eigen::Matrix<T, state_size, 1>;

template <int action_size, typename T = double>
using EigenAction = Eigen::Matrix<T, action_size, 1>;

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

struct SE2 {
    double x;
    double y;
    double theta;
};

} // namespace mpex