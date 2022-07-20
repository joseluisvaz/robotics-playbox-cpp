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

#include <Eigen/Dense>
#include <array>
#include <easy/profiler.h>
#include <random>
#include <thread>

namespace Magnum::Examples
{
template <typename T, typename A>
int arg_min(std::vector<T, A> const &vec)
{
  return static_cast<int>(std::distance(vec.begin(), std::min_element(vec.begin(), vec.end())));
}

using namespace Eigen;

constexpr int state_size = 6;
constexpr int action_size = 2;
using EigenState = Vector<float, state_size>;
using EigenAction = Vector<float, action_size>;

using EigenStateSequence = Matrix<float, state_size, Dynamic>;
using EigenActionSequence = Matrix<float, action_size, Dynamic>;

struct EigenTrajectory
{
  std::vector<float> times{};
  EigenStateSequence states;
  EigenActionSequence actions;
};

struct EigenKinematicBicycle
{
  constexpr static float ts{0.1};
  constexpr static float one_over_wheelbase{1.0 / 3.0};
  constexpr static float max_steering{0.52};
  constexpr static float max_steering_rate{0.1};
  constexpr static float max_jerk{0.6};

  static void step(const Ref<const EigenState> &state, const Ref<const EigenAction> &action, Ref<EigenState> new_state)
  {
    new_state[0] = state[0] + ts * state[3] * std::cos(state[2]);
    new_state[1] = state[1] + ts * state[3] * std::sin(state[2]);
    new_state[2] =
        std::clamp(state[2] + ts * state[3] * one_over_wheelbase * std::tan(state[5]), -max_steering, max_steering);
    new_state[3] = state[3] + ts * state[4];
    new_state[4] = state[4] + ts * max_jerk * std::tanh(action[0]);
    new_state[5] = state[5] + ts * max_steering_rate * std::tanh(action[1]);
  }
};

struct CostFunction
{

  float evaluate_state_action_pair(const Ref<const EigenState> &state, const Ref<const EigenAction> &action) const
  {

    return 1.0f * (state[3] - 3.0f) * (state[3] - 3.0f) + 10000.0f * (state[2] * state[2]) +
           1.0f * (state[4] * state[4]) + 1.0f * (state[5] - state[5]) + 100.0f * (action[0] - action[0]) +
           100.0f * (action[1] - action[1]);
  }

  float operator()(const Ref<const EigenStateSequence> &states, const Ref<const EigenActionSequence> &actions) const
  {
    float cost = 0.0f;
    for (int i{0}; i < states.cols(); ++i)
    {
      cost += evaluate_state_action_pair(states.col(i), actions.col(i));
    }
    return cost;
  }
};

struct NormalRandomVariable
{
  NormalRandomVariable() = default;
  NormalRandomVariable(const EigenActionSequence &mean, const EigenActionSequence &covar)
      : mean(mean), transform(covar){};

  EigenActionSequence mean;
  EigenActionSequence transform;

  EigenActionSequence operator()() const
  {
    static std::mt19937 gen{std::random_device{}()};
    static std::normal_distribution<float> dist;

    return mean.array() + transform.array() * (EigenActionSequence::Zero(mean.rows(), mean.cols())
                                                   .unaryExpr([&](auto x) { return dist(gen); })
                                                   .array());
  }
};

template <typename DynamicsT>
class CEM_MPC
{

public:
  CEM_MPC() = default;
  CEM_MPC(const int num_iters, const int horizon, const int population, const int elites)
      : num_iters_(num_iters), horizon_(horizon), population_(population), elites_(elites)
  {
    trajectory_.states = EigenStateSequence::Zero(state_size, horizon_);
    trajectory_.actions = EigenActionSequence::Zero(action_size, horizon_);
    trajectory_.times = std::vector<float>(horizon_, 0.0f);

    costs_index_pair_ = std::vector<std::pair<float, int>>(population_, std::make_pair(0.0f, 0));

    for (int i = 0; i < population_; ++i)
    {
      EigenTrajectory trajectory;
      trajectory.states = EigenStateSequence::Zero(state_size, horizon_);
      trajectory.actions = EigenActionSequence::Zero(action_size, horizon_);
      trajectory.times = std::vector<float>(horizon_, 0.0f);
      candidate_trajectories_.emplace_back(std::move(trajectory));
    }

    EigenActionSequence mean = EigenActionSequence::Zero(trajectory_.actions.rows(), trajectory_.actions.cols());
    EigenActionSequence stddev = EigenActionSequence::Ones(trajectory_.actions.rows(), trajectory_.actions.cols());

    sampler_ = NormalRandomVariable(mean, stddev);
  };

  void rollout(EigenTrajectory &trajectory)
  {
    EASY_FUNCTION(profiler::colors::Green);
    double current_time_s{0.0};
    for (size_t i = 0; i + 1 < horizon_; ++i)
    {
      trajectory.times.at(i) = current_time_s;

      DynamicsT::step(trajectory.states.col(i), trajectory.actions.col(i), trajectory.states.col(i + 1));
      current_time_s += DynamicsT::ts;
    }
    trajectory.times.at(horizon_ - 1) = current_time_s;
  }

  void run_cem_iteration(const Ref<EigenState> &initial_state)
  {
    EASY_FUNCTION(profiler::colors::Yellow);
    CostFunction cost_function;

    for (int j = 0; j < population_; j++)
    {
      auto &trajectory = candidate_trajectories_.at(j);
      trajectory.states.col(0) = initial_state; // Set first state
      trajectory.actions = sampler_();          // Sample actions
      rollout(trajectory);

      costs_index_pair_.at(j).first = cost_function(trajectory.states, trajectory.actions);
      costs_index_pair_.at(j).second = j;
    }

    std::sort(costs_index_pair_.begin(), costs_index_pair_.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    EigenActionSequence mean_actions = EigenActionSequence::Zero(action_size, horizon_);
    for (int i = 0; i < elites_; ++i)
    {
      const auto &elite_index = costs_index_pair_.at(i).second;
      mean_actions += candidate_trajectories_.at(elite_index).actions;
    }

    mean_actions = mean_actions / elites_;

    EigenActionSequence stddev = EigenActionSequence::Zero(action_size, horizon_);
    for (int i = 0; i < elites_; ++i)
    {
      const auto &elite_index = costs_index_pair_.at(i).second;
      EigenActionSequence temp = candidate_trajectories_.at(elite_index).actions - mean_actions;
      stddev = stddev + temp.cwiseProduct(temp);
    }

    stddev = (stddev / elites_).cwiseSqrt();

    sampler_.mean = mean_actions;
    sampler_.transform = stddev;
  }

  EigenTrajectory &execute(const Ref<EigenState> &initial_state)
  {
    EASY_FUNCTION(profiler::colors::Magenta);

    sampler_.mean = EigenActionSequence::Zero(trajectory_.actions.rows(), trajectory_.actions.cols());
    sampler_.transform = EigenActionSequence::Ones(trajectory_.actions.rows(), trajectory_.actions.cols());

    // std::vector<std::thread> my_threads;

    for (int i = 0; i < num_iters_; ++i)
    {
      run_cem_iteration(initial_state);
      // my_threads.push_back(std::thread([](int i) { printf("this is the number : %d\n", i); }, 1059));
    }

    // for (auto &thread : my_threads)
    // {
    //   thread.join();
    // }

    trajectory_.states.col(0) = initial_state; // Set first state

    trajectory_.actions = sampler_.mean; // Set mean actions
    rollout(trajectory_);
    return trajectory_;
  };

private:
  int num_iters_;
  int horizon_;
  int population_;
  int elites_;
  std::vector<EigenTrajectory> candidate_trajectories_;
  EigenTrajectory trajectory_;
  EigenTrajectory final_trajectory;
  NormalRandomVariable sampler_;

  std::vector<std::pair<float, int>> costs_index_pair_;
};

} // namespace Magnum::Examples