#pragma once

#include "ilqr_mpc.hpp"
#include "finite_diff.hpp"
#include "math.hpp"
#include "types.hpp"
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <array>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <easy/profiler.h>
#include <stdexcept>

namespace RoboticsSandbox
{
using namespace Eigen;

namespace
{

autodiff::dual2nd cost_function_diff(VectorXdual2nd x, VectorXdual2nd u)
{
  VectorXdual2nd x_ref = VectorXdual2nd::Zero(x.size());
  x_ref(3) = 0.0;
  x_ref(0) = 100.0;
  x_ref(1) = 0.0;

  VectorXdual2nd Q(x.size());
  Q << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;
  VectorXdual2nd R(u.size());
  R << 0.1, 0.1;

  return ((x - x_ref).transpose() * Q.asDiagonal() * (x - x_ref) + u.transpose() * R.asDiagonal() * u).value();
}

} // namespace

iLQR_MPC::iLQR_MPC(const int horizon, const int iters) : horizon_(horizon), iters_(iters)
{
  initialize_matrices(horizon_);
}

void iLQR_MPC::plot_trajectory(const Trajectory &trajectory)
{
  vector<double> x_vals;
  vector<double> y_vals;
  for (size_t i = 0; i + 1 < static_cast<size_t>(horizon_ + 1); ++i)
  {
    x_vals.push_back(trajectory.states(0, i));
    y_vals.push_back(trajectory.states(1, i));
  }
  plt::plot(x_vals, y_vals);
}

void iLQR_MPC::rollout(Trajectory &trajectory)
{
  auto &u = trajectory.actions;
  auto &x = trajectory.states;

  u = Actions::Zero(trajectory.actions.rows(), trajectory.actions.cols());

  double current_time_s{0.0f};
  for (size_t i = 0; i + 1 < static_cast<size_t>(horizon_ + 1); ++i)
  {
    trajectory.times.at(i) = current_time_s;
    DynamicsT::step(x.col(i), u.col(i), x.col(i + 1));
    current_time_s += DynamicsT::ts;
  }
  trajectory.times.at(horizon_) = current_time_s;
}

iLQR_MPC::Trajectory iLQR_MPC::solve(const Ref<State> &x0)
{
  EASY_FUNCTION(profiler::colors::Grey);
  auto trajectory = Trajectory(horizon_ + 1);
  this->rollout(trajectory);

  cout.precision(17);
  for (int i{0}; i < iters_; ++i)
  {
    this->backward_pass(trajectory);
    auto cost = compute_cost(trajectory);
    cout << "iter: " << i << " cost: " << std::fixed << cost << endl;

    // Do backtracking line search to find when the cost decreases the most, forward passes are inexpesive,
    // so this it is ok to iterate for maximum of "backtracking_iterations".
    constexpr int backtracking_iterations = 10;
    for (int j{0}; j < backtracking_iterations; ++j)
    {
      // The value of alpha will be decreased exponentially with each iteration to find a decreasing cost.
      auto new_trajectory = this->forward_pass(trajectory, /* alpha= */ pow(0.5, j));
      auto new_cost = compute_cost(new_trajectory);

      if (std ::abs((cost - new_cost) / (cost + 1e-4)) < tol_)
      {
        // First check for convergence
        return new_trajectory;
      }

      if (new_cost < cost)
      {
        // If the cost decreased then update the trajectory, and break the loop to continue the main ilqr loop.
        trajectory = new_trajectory;
        cost = new_cost;
        break;
      }
    }
  }
  return trajectory;
}

iLQR_MPC::Trajectory iLQR_MPC::forward_pass(const Trajectory &trajectory, const double alpha)
{
  EASY_FUNCTION(profiler::colors::Blue);
  Trajectory new_trajectory = Trajectory(horizon_ + 1);
  auto &u = new_trajectory.actions;
  auto &u_bar = trajectory.actions;
  auto &x = new_trajectory.states;
  auto &x_bar = trajectory.states;

  x.col(0) = x_bar.col(0);

  // Compute to horizon_ to account for the terminal state
  for (size_t i = 0; i + 1 < static_cast<size_t>(horizon_ + 1); ++i)
  {
    u.col(i) = u_bar.col(i) + alpha * k[i] + K[i] * (x.col(i) - x_bar.col(i));
    DynamicsT::step(x.col(i), u.col(i), x.col(i + 1));
  }
  return new_trajectory;
}

double iLQR_MPC::compute_cost(const Trajectory &trajectory)
{
  double cost = 0.0;
  for (int i{0}; i < horizon_; ++i)
  {
    cost += cost_function_diff(trajectory.states.col(i), trajectory.actions.col(i)).val.val;
  }

  const int n = horizon_;
  Vector zero_action = Vector::Zero(trajectory.actions.rows());
  cost += cost_function_diff(trajectory.states.col(n), zero_action).val.val;
  return cost;
}

void iLQR_MPC::backward_pass(const Trajectory &trajectory)
{
  EASY_FUNCTION(profiler::colors::Green);
  this->compute_derivatives(trajectory);

  Vector Vx = lx.back();
  Matrix Vxx = lxx.back();

  Matrix I = Matrix::Identity(DynamicsT::state_size, DynamicsT::state_size);

  for (int i = horizon_ - 1; i >= 0; --i)
  {
    const auto &Qx = lx[i] + (fx[i].transpose() * Vx);
    const auto &Qu = lu[i] + (fu[i].transpose() * Vx);

    const auto &Qxx = lxx[i] + (fx[i].transpose() * Vxx * fx[i]);
    const auto &Quu = luu[i] + fu[i].transpose() * (Vxx + (mu_ * I)) * fu[i];
    const auto &Qux = lux[i] + fu[i].transpose() * (Vxx + (mu_ * I)) * fx[i];

    for (int j{0}; j < K[i].cols(); ++j)
    {
      K[i].col(j) = -Quu.ldlt().solve(Qux.col(j));
    }
    k[i] = -Quu.ldlt().solve(Qu);

    Vx = Qx + K[i].transpose() * Quu * k[i] + K[i].transpose() * Qu + Qux.transpose() * k[i];
    Vxx = Qxx + K[i].transpose() * Quu * K[i] + K[i].transpose() * Qux + Qux.transpose() * K[i];
    Vxx = 0.5 * (Vxx + Vxx.transpose());
  }
}

void iLQR_MPC::compute_derivatives(const Trajectory &trajectory)
{
  VectorXdual2nd x; // placeholder for the current state as differentiable type
  VectorXdual2nd u; // placeholder for the current action as differentiable type

  dual2nd cost;                 // unused
  Vector state_action_gradient; // placeholder for the full state_action_gradient
  Matrix state_action_hessian;  // placeholder for the full state_action_hessian

  for (int i = 0; i < horizon_; ++i)
  {
    x = trajectory.states.col(i);
    u = trajectory.actions.col(i);

    fx[i] = jacobian(DynamicsT::step_diff<VectorXdual2nd>, wrt(x), at(x, u));
    fu[i] = jacobian(DynamicsT::step_diff<VectorXdual2nd>, wrt(u), at(x, u));

    // Compute full hessian and full gradient
    state_action_hessian = hessian(cost_function_diff, wrt(x, u), at(x, u), cost, state_action_gradient);

    // Split full state_action_gradient and hessian into individual matrices
    lx[i] = state_action_gradient.head(state_size);
    lu[i] = state_action_gradient.tail(action_size);
    lxx[i] = state_action_hessian.topLeftCorner(state_size, state_size);
    luu[i] = state_action_hessian.bottomRightCorner(action_size, action_size);
    lux[i] = state_action_hessian.bottomLeftCorner(action_size, state_size);
  }

  // Final cost hessian and gradient calculation
  x = trajectory.states.col(horizon_ - 1);
  u = trajectory.actions.col(horizon_ - 1);
  lxx[horizon_] = hessian(cost_function_diff, wrt(x), at(x, u), cost, lx[horizon_]);
}

void iLQR_MPC::initialize_matrices(const int horizon)
{
  const int state_size = DynamicsT::state_size;
  const int action_size = DynamicsT::action_size;

  fx.resize(horizon_);
  fu.resize(horizon_);
  lx.resize(horizon_ + 1);
  lu.resize(horizon_);
  lxx.resize(horizon_ + 1);
  luu.resize(horizon_);
  lux.resize(horizon_);

  K.resize(horizon_);
  k.resize(horizon_);

  std::fill(fx.begin(), fx.end(), Matrix::Zero(state_size, state_size));
  std::fill(fu.begin(), fu.end(), Matrix::Zero(action_size, action_size));
  std::fill(lx.begin(), lx.end(), Vector::Zero(state_size));
  std::fill(lu.begin(), lu.end(), Vector::Zero(action_size));
  std::fill(lxx.begin(), lxx.end(), Matrix::Zero(state_size, state_size));
  std::fill(luu.begin(), luu.end(), Matrix::Zero(action_size, action_size));
  std::fill(lux.begin(), lux.end(), Matrix::Zero(action_size, state_size));
  std::fill(K.begin(), K.end(), Matrix::Zero(action_size, state_size));
  std::fill(k.begin(), k.end(), Vector::Zero(action_size));
}

} // namespace RoboticsSandbox
