#pragma once

#include "math.hpp"
#include "types.hpp"
#include <Eigen/Dense>
#include <array>
#include <thread>

namespace RoboticsSandbox
{
using namespace Eigen;

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
  ///@param num_iters The number of iterations of the CEM method.
  ///@param horizon The number of points in the horizon.
  ///@param population  The population size per iteration.
  ///@param elites The number of samples elite samples to use per iteration. Must be less than population.
  CEM_MPC(const int num_iters, const int horizon, const int population, const int elites);

  /// Execute the cross entropy method mpc, it return the full state-action trajectory. Extract the first action to
  /// control your system.
  ///@param[in] initial_state The initial state
  ///@return Trajectory& A reference to the trajectory that the MPC computed.
  Trajectory &execute(const Ref<State> &initial_state);

  int &get_num_iters_mutable();

  const Trajectory &get_trajectory() const;

  CostFunction cost_function_;

  std::vector<Trajectory> candidate_trajectories_;

private:
  /// Runs a single rollout for a trajectory.
  ///@param[in, out] trajectory The modified trajectory after a rollout.
  void rollout(Trajectory &trajectory);

  /// Runs a full iteration of the CEM method using multiple threads, parallelizing the rollouts.
  ///@param[in] initial_state The initial state of the system.
  void run_cem_iteration(const Ref<State> &initial_state);

  /// Runs a full iteration of the CEM method using multiple threads, parallelizing the rollouts.
  void update_action_distribution();

  int num_iters_;
  int horizon_;
  int population_;
  int elites_;

  Trajectory trajectory_;
  Sampler sampler_;

  std::vector<std::thread> threads_{16};
  std::vector<std::pair<float, int>> costs_index_pair_;
};

} // namespace RoboticsSandbox
