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

double cost_function(Vector x, Vector u)
{
  Vector x_ref = Vector::Zero(x.size());
  x_ref(3) = 0.0;
  x_ref(0) = 100.0;
  x_ref(1) = 0.0;
  // x_ref(1) = 13.0;

  Vector Q(x.size());
  Q << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;
  Vector R(u.size());
  R << 0.1, 0.1;

  double result = ((x - x_ref).transpose() * Q.asDiagonal() * (x - x_ref) + u.transpose() * R.asDiagonal() * u).value();
  return result + std::sqrt(2);
}

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

// TODO: make it a free function
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

  sampler_.mean_ = Actions::Zero(trajectory.actions.rows(), trajectory.actions.cols());
  sampler_.stddev_ = Actions::Ones(trajectory.actions.rows(), trajectory.actions.cols());

  u = this->sampler_();

  double current_time_s{0.0f};
  for (size_t i = 0; i + 1 < static_cast<size_t>(horizon_ + 1); ++i)
  {
    trajectory.times.at(i) = current_time_s;
    // u(0, i) += 0.1;
    // u(1, i) += 0.01;
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
    double J = compute_cost(trajectory);
    cout << "iter: " << i << " cost: " << std::fixed << J << endl;
    // plot_trajectory(trajectory);

    for (int j{0}; j < 10; ++j) // Iterate over alphas
    {
      auto new_trajectory = this->forward_pass(trajectory, /* alpha= */ pow(0.5, j));
      double J_new = compute_cost(new_trajectory);
      if (J_new < J)
      {
        trajectory = new_trajectory;
        if (std ::abs((J - J_new) / (J + 1e-4)) < tol_)
        {
          return trajectory;
        }

        J = J_new;
        break;
      }
    }
  }
  // plt::show();

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
  this->compute_matrices(trajectory);

  Vector Vx = lx.back();
  Matrix Vxx = lxx.back();

  Matrix I = Matrix::Identity(DynamicsT::state_size, DynamicsT::state_size);

  for (int i = horizon_ - 1; i >= 0; --i)
  {
    auto Qx = lx[i] + (fx[i].transpose() * Vx);
    auto Qu = lu[i] + (fu[i].transpose() * Vx);

    auto Qxx = lxx[i] + (fx[i].transpose() * Vxx * fx[i]);
    auto Quu = luu[i] + fu[i].transpose() * (Vxx + (mu_ * I)) * fu[i];
    auto Qux = lux[i] + fu[i].transpose() * (Vxx + (mu_ * I)) * fx[i];

    // Matrix Quu_inv = Quu.inverse();
    // K[i] = -Quu_inv * Qux;
    // k[i] = -Quu_inv * Qu;
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

// void iLQR_MPC::compute_matrices(const Trajectory &trajectory)
// {
//   EASY_FUNCTION(profiler::colors::Yellow);
//   for (int i = 0; i < horizon_; ++i)
//   {
//     VectorXdual2nd x = trajectory.states.col(i);
//     VectorXdual2nd u = trajectory.actions.col(i);
//
//     fx[i] = calc_jacobian_x(DynamicsT::step_, trajectory.states.col(i), trajectory.actions.col(i));
//     fu[i] = calc_jacobian_u(DynamicsT::step_, trajectory.states.col(i), trajectory.actions.col(i));
//     lx[i] = calc_grad_x(cost_function, trajectory.states.col(i), trajectory.actions.col(i));
//     lu[i] = calc_grad_u(cost_function, trajectory.states.col(i), trajectory.actions.col(i), 0.01);
//     lxx[i] = calc_hessian_x(cost_function, trajectory.states.col(i), trajectory.actions.col(i), 0.01);
//     luu[i] = calc_hessian_u(cost_function, trajectory.states.col(i), trajectory.actions.col(i), 0.01);
//     lux[i] = calc_hessian_ux(cost_function, trajectory.states.col(i), trajectory.actions.col(i), 0.01);
//   }
//
//   const int T = horizon_;
//   lx[T] = calc_grad_x(cost_function, trajectory.states.col(T - 1), trajectory.actions.col(T - 1));
//   lxx[T] = calc_hessian_x(cost_function, trajectory.states.col(T - 1), trajectory.actions.col(T - 1));
// }

void iLQR_MPC::compute_matrices(const Trajectory &trajectory)
{
  EASY_FUNCTION(profiler::colors::Yellow);

  VectorXdual2nd x;
  VectorXdual2nd u;

  dual2nd cost; // unused
  Vector G;
  Matrix H;

  for (int i = 0; i < horizon_; ++i)
  {
    x = trajectory.states.col(i);
    u = trajectory.actions.col(i);

    fx[i] = jacobian(DynamicsT::step_diff<VectorXdual2nd>, wrt(x), at(x, u));
    fu[i] = jacobian(DynamicsT::step_diff<VectorXdual2nd>, wrt(u), at(x, u));

    H = hessian(cost_function_diff, wrt(x, u), at(x, u), cost, G);

    lx[i] = G.head(state_size);
    lu[i] = G.tail(action_size);
    lxx[i] = H.topLeftCorner(state_size, state_size);
    luu[i] = H.bottomRightCorner(action_size, action_size);
    lux[i] = H.bottomLeftCorner(action_size, state_size);
  }

  const int T = horizon_;

  x = trajectory.states.col(T - 1);
  u = trajectory.actions.col(T - 1);

  lxx[T] = hessian(cost_function_diff, wrt(x), at(x, u), cost, lx[T]);
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
