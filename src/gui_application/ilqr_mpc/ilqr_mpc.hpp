#pragma once

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <array>
#include <optional>

#include "common/dynamics.h"
#include "common/math.hpp"
#include "common/types.hpp"
#include "third_party/matplotlibcpp.h"
namespace mpex
{
using namespace Eigen;

namespace plt = matplotlibcpp;

/* Implementation of a simple Cross Entropy Method Model Predictive Control (CEM-MPC) */
class IterativeLinearQuadraticRegulator
{
  using DynamicsT = EigenKinematicBicycle;

  using State = typename DynamicsT::State;
  using Action = typename DynamicsT::Action;
  using States = typename DynamicsT::States;
  using Actions = typename DynamicsT::Actions;
  using Trajectory = typename DynamicsT::Trajectory;
  using Sampler = NormalRandomVariable<Actions>;

  constexpr static int state_size = DynamicsT::state_size;
  constexpr static int action_size = DynamicsT::action_size;

  template <typename EigenT>
  using VectorOfEigen = std::vector<EigenT, Eigen::aligned_allocator<EigenT>>;

  void initialize_matrices(const int horizon);
  void compute_derivatives(const Trajectory &trajectory);
  void plot_trajectory(const Trajectory &trajectory);

public:
  IterativeLinearQuadraticRegulator() = default;
  IterativeLinearQuadraticRegulator(const int horizon, const int iters, const bool debug);

  Trajectory solve(const Ref<State> &x0, const std::optional<Trajectory>& maybe_warmstart = {});

  int horizon_;

private:
  /// Runs a single rollout for a trajectory.
  ///@param[in, out] trajectory The modified trajectory after a rollout.
  ///@param[in] use_warmstart Whether to use warmstart 
  void rollout(Trajectory &trajectory, const bool use_warmstart);
  void backward_pass(const Trajectory &trajectory);
  Trajectory forward_pass(const Trajectory &trajectory, const double alpha);
  double compute_cost(const Trajectory &trajectory);

  int iters_;
  double mu_{0.5};
  double tol_{1e-5};
  bool debug_;

  // Instantiate sampler for initial action distribution
  Sampler sampler_;

  // Derivatives of the dynamics
  VectorOfEigen<Matrix> fx; // [s, s] length (n)
  VectorOfEigen<Matrix> fu; // [s, a] length (n)

  // Derivatives of the cost function
  VectorOfEigen<Vector> lx;  // [s] length (n + 1)
  VectorOfEigen<Matrix> lxx; // [s, s] length (n + 1)
  VectorOfEigen<Vector> lu;  // [a] length (n)
  VectorOfEigen<Matrix> luu; // [a, a] length (n)
  VectorOfEigen<Matrix> lux; // [a, s] length (n)

  // Control Matrices
  VectorOfEigen<Matrix> K; // [a, s] length (n)
  VectorOfEigen<Vector> k; // [a] length (n)
};

} // namespace mpex
