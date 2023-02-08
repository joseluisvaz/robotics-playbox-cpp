#pragma once

#include <array>
#include <memory>
#include <optional>
#include <thread>

#include <Eigen/Dense>

#include "common/math.hpp"
#include "common/types.hpp"
#include "cost_function.hpp"

namespace mpex {
using namespace Eigen;

struct CEM_MPC_Config
{
    /// The number of iterations of the CEM method.
    int num_iters = 10;
    /// The number of points in the horizon.
    int horizon = 20;
    /// The population size per iteration.
    int population = 128;
    /// The number of samples elite samples to use per iteration. Must be less than population.
    int elites = 2;
};

/* Implementation of a simple Cross Entropy Method Model Predictive Control (CEM-MPC) */
template <typename DynamicsT>
class CEM_MPC
{
    using State = typename DynamicsT::State;
    using Action = typename DynamicsT::Action;
    using States = typename DynamicsT::States;
    using Actions = typename DynamicsT::Actions;
    using Trajectory = typename DynamicsT::Trajectory;
    using Sampler = NormalRandomVariable<Actions>;
    constexpr static int state_size = DynamicsT::state_size;
    constexpr static int action_size = DynamicsT::action_size;

  public:
    CEM_MPC() = default;

    /// Construct a new cem_mpc object, preallocates memory for trajectories.
    CEM_MPC(const CEM_MPC_Config &config, const std::shared_ptr<DynamicsT> dynamics_ptr_);

    /// Execute the cross entropy method mpc, it return the full state-action trajectory. Extract the first action to
    /// control your system.
    ///@param[in] initial_state The initial state
    ///@param[in] maybe_trajectory The initial trajectory containing the initial control sequence for warm start
    ///@return Trajectory& A reference to the trajectory that the MPC computed.
    Trajectory solve(const Ref<State> &initial_state, const std::optional<Trajectory> &maybe_trajectory);

    const CEM_MPC_Config &get_config() const;
    CEM_MPC_Config &get_config_mutable();
    const Trajectory &get_last_solution() const;

    std::shared_ptr<CostFunction> cost_function_;
    std::vector<Trajectory> candidate_trajectories_;
    std::vector<std::pair<double, int>> costs_index_pair_;

    // Dynamics function pointer
    std::shared_ptr<DynamicsT> dynamics_ptr_;

  private:
    /// Runs a single rollout for a trajectory.
    ///@param[in, out] trajectory The modified trajectory after a rollout.
    void rollout(Trajectory &trajectory);

    /// Runs a full iteration of the CEM method using multiple threads, parallelizing the rollouts.
    ///@param[in] initial_state The initial state of the system.
    void run_cem_iteration(const Ref<State> &initial_state);

    /// Runs a full iteration of the CEM method using multiple threads, parallelizing the rollouts.
    void update_action_distribution();

    std::vector<std::thread> threads_{16};
    CEM_MPC_Config config_;
    Trajectory trajectory_;
    Sampler sampler_;
};

} // namespace mpex
