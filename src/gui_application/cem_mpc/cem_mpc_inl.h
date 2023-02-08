#pragma once

#include <iostream>
#include <memory>
#include <random>
#include <thread>

#include <Eigen/Dense>
#include <easy/profiler.h>

#include "cem_mpc/cem_mpc.h"
#include "common/math.hpp"
#include "common/types.hpp"

namespace mpex {

template <typename DynamicsT>
CEM_MPC<DynamicsT>::CEM_MPC(const CEM_MPC_Config &config, const std::shared_ptr<DynamicsT> dynamics_ptr)
    : config_(config), dynamics_ptr_(dynamics_ptr), trajectory_(config.horizon)
{
    costs_index_pair_ = std::vector<std::pair<double, int>>(config.population, std::make_pair(0.0f, 0));

    for (int i = 0; i < config.population; ++i)
    {
        candidate_trajectories_.emplace_back(Trajectory(config.horizon));
    }

    sampler_.mean_ = Actions::Zero(trajectory_.actions.rows(), trajectory_.actions.cols());
    sampler_.stddev_ = Actions::Ones(trajectory_.actions.rows(), trajectory_.actions.cols());
}

template <typename DynamicsT>
void CEM_MPC<DynamicsT>::rollout(Trajectory &trajectory)
{
    double current_time_s{0.0f};
    for (size_t i = 0; i + 1 < static_cast<size_t>(config_.horizon); ++i)
    {
        trajectory.times.at(i) = current_time_s;

        Vector parameters = Vector::Zero(5);
        dynamics_ptr_->step(trajectory.states.col(i), trajectory.actions.col(i), parameters, trajectory.states.col(i + 1));
        current_time_s += DynamicsT::ts;
    }
    trajectory.times.at(config_.horizon - 1) = current_time_s;
}

template <typename DynamicsT>
void CEM_MPC<DynamicsT>::run_cem_iteration(const Ref<State> &initial_state)
{
    EASY_FUNCTION(profiler::colors::Yellow);

    /// Helper function to run multithreaded rollouts.
    const auto evaluate_trajectory_fn = [this, &initial_state](int k) {
        int block_size = static_cast<int>(config_.population / this->threads_.size());
        for (int j = block_size * k; j < block_size * k + block_size; ++j)
        {
            auto &trajectory = this->candidate_trajectories_.at(j);
            trajectory.states.col(0) = initial_state; // Set first state
            trajectory.actions = this->sampler_();    // Sample actions
            this->rollout(trajectory);

            // keep track of the cost and the index.
            this->costs_index_pair_.at(j).first = (*(this->cost_function_))(trajectory.states, trajectory.actions);
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
    auto &candidate_trajectories = candidate_trajectories_;

    //  Sort the cost index pairs to get the best trajectories at the beginning.
    std::sort(costs_index_pair_.begin(), costs_index_pair_.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    // Compute the mean actions using best trajectories.
    Actions mean_actions = Actions::Zero(action_size, config_.horizon);
    for (int i = 0; i < config_.elites; ++i)
    {
        const auto &elite_index = costs_index_pair_.at(i).second;
        mean_actions += candidate_trajectories.at(elite_index).actions;
    }

    mean_actions = mean_actions / config_.elites;

    // Compute the stddev of the actions using the best trajectories.
    Actions stddev = Actions::Zero(action_size, config_.horizon);
    for (int i = 0; i < config_.elites; ++i)
    {
        const auto &elite_index = costs_index_pair_.at(i).second;
        Actions temp = candidate_trajectories.at(elite_index).actions - mean_actions;
        stddev = stddev + temp.cwiseProduct(temp);
    }

    stddev = (stddev / config_.elites).cwiseSqrt();

    sampler_.mean_ = mean_actions;
    sampler_.stddev_ = stddev;
}

template <typename DynamicsT>
typename CEM_MPC<DynamicsT>::Trajectory CEM_MPC<DynamicsT>::solve(
    const Ref<State> &initial_state, const std::optional<Trajectory> &maybe_trajectory)
{
    EASY_FUNCTION(profiler::colors::Magenta);

    sampler_.mean_ =
        maybe_trajectory.has_value() ? maybe_trajectory->actions : Actions::Zero(trajectory_.actions.rows(), trajectory_.actions.cols());
    sampler_.stddev_ = Actions::Ones(trajectory_.actions.rows(), trajectory_.actions.cols());

    for (int i = 0; i < config_.num_iters; ++i)
    {
        run_cem_iteration(initial_state);
    }

    trajectory_.states.col(0) = initial_state; // Set first state
    trajectory_.actions = sampler_.mean_;      // Set mean actions
    rollout(trajectory_);
    return trajectory_;
}

template <typename DynamicsT>
const CEM_MPC_Config &CEM_MPC<DynamicsT>::get_config() const
{
    return config_;
}

template <typename DynamicsT>
CEM_MPC_Config &CEM_MPC<DynamicsT>::get_config_mutable()
{
    return config_;
}

template <typename DynamicsT>
const typename DynamicsT::Trajectory &CEM_MPC<DynamicsT>::get_last_solution() const
{
    return trajectory_;
}

} // namespace mpex