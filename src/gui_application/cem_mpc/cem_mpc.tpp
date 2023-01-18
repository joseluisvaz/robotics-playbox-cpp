#pragma once

#include <Eigen/Dense>
#include <easy/profiler.h>
#include <iostream>
#include <random>
#include <thread>

#include "cem_mpc/cem_mpc.hpp"
#include "common/math.hpp"
#include "common/types.hpp"

namespace RoboticsSandbox
{

template <typename DynamicsT>
CEM_MPC<DynamicsT>::CEM_MPC(const int num_iters, const int horizon, const int population, const int elites)
    : num_iters_(num_iters), horizon_(horizon), population_(population), elites_(elites), trajectory_(horizon)
{
  costs_index_pair_ = std::vector<std::pair<float, int>>(population_, std::make_pair(0.0f, 0));

  for (int i = 0; i < population_; ++i)
  {
    candidate_trajectories_.emplace_back(Trajectory(horizon_));
  }

  sampler_.mean_ = Actions::Zero(trajectory_.actions.rows(), trajectory_.actions.cols());
  sampler_.stddev_ = Actions::Ones(trajectory_.actions.rows(), trajectory_.actions.cols());
}

template <typename DynamicsT>
void CEM_MPC<DynamicsT>::rollout(Trajectory &trajectory)
{
  float current_time_s{0.0f};
  for (size_t i = 0; i + 1 < static_cast<size_t>(horizon_); ++i)
  {
    trajectory.times.at(i) = current_time_s;

    DynamicsT::step(trajectory.states.col(i), trajectory.actions.col(i), trajectory.states.col(i + 1));
    current_time_s += DynamicsT::ts;
  }
  trajectory.times.at(horizon_ - 1) = current_time_s;
}

template <typename DynamicsT>
void CEM_MPC<DynamicsT>::run_cem_iteration(const Ref<State> &initial_state)
{
  EASY_FUNCTION(profiler::colors::Yellow);

  /// Helper function to run multithreaded rollouts.
  const auto evaluate_trajectory_fn = [this, &initial_state](int k)
  {
    int block_size = static_cast<int>(this->population_ / this->threads_.size());
    for (int j = block_size * k; j < block_size * k + block_size; ++j)
    {
      auto &trajectory = this->candidate_trajectories_.at(j);
      trajectory.states.col(0) = initial_state; // Set first state
      trajectory.actions = this->sampler_();    // Sample actions
      this->rollout(trajectory);

      // keep track of the cost and the index.
      this->costs_index_pair_.at(j).first = this->cost_function_(trajectory.states, trajectory.actions);
      this->costs_index_pair_.at(j).second = j;
    }
  };

  for (size_t k = 0; k < threads_.size(); k++)
  {
    threads_.at(k) = std::thread([&evaluate_trajectory_fn, k]() { evaluate_trajectory_fn(k); });
  }

  for (auto &&thread : threads_)
  {
    if (thread.joinable())
    {
      thread.join();
    }
  }

  update_action_distribution();
}

template <typename DynamicsT>
void CEM_MPC<DynamicsT>::update_action_distribution()
{
  //  Sort the cost index pairs to get the best trajectories at the beginning.
  std::sort(costs_index_pair_.begin(), costs_index_pair_.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

  // Compute the mean actions using best trajectories.
  Actions mean_actions = Actions::Zero(action_size, horizon_);
  for (int i = 0; i < elites_; ++i)
  {
    const auto &elite_index = costs_index_pair_.at(i).second;
    mean_actions += candidate_trajectories_.at(elite_index).actions;
  }

  mean_actions = mean_actions / elites_;

  // Compute the stddev of the actions using the best trajectories.
  Actions stddev = Actions::Zero(action_size, horizon_);
  for (int i = 0; i < elites_; ++i)
  {
    const auto &elite_index = costs_index_pair_.at(i).second;
    Actions temp = candidate_trajectories_.at(elite_index).actions - mean_actions;
    stddev = stddev + temp.cwiseProduct(temp);
  }

  stddev = (stddev / elites_).cwiseSqrt();

  sampler_.mean_ = mean_actions;
  sampler_.stddev_ = stddev;
}

template <typename DynamicsT>
typename CEM_MPC<DynamicsT>::Trajectory &CEM_MPC<DynamicsT>::execute(const Ref<State> &initial_state)
{
  EASY_FUNCTION(profiler::colors::Magenta);

  sampler_.mean_ = Actions::Zero(trajectory_.actions.rows(), trajectory_.actions.cols());
  sampler_.stddev_ = Actions::Ones(trajectory_.actions.rows(), trajectory_.actions.cols());

  for (int i = 0; i < num_iters_; ++i)
  {
    run_cem_iteration(initial_state);
  }

  trajectory_.states.col(0) = initial_state; // Set first state
  trajectory_.actions = sampler_.mean_;      // Set mean actions
  rollout(trajectory_);
  return trajectory_;
}

template <typename DynamicsT>
int &CEM_MPC<DynamicsT>::get_num_iters_mutable()
{
  return num_iters_;
}

template <typename DynamicsT>
const typename DynamicsT::Trajectory &CEM_MPC<DynamicsT>::get_trajectory() const
{
  return trajectory_;
}

} // namespace RoboticsSandbox