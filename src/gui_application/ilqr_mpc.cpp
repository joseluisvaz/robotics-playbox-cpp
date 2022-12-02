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

namespace RoboticsSandbox
{
using namespace Eigen;

namespace
{

double cost_function(Vector x, Vector u)
{
  Vector x_ref = Vector::Zero(x.size());
  x_ref(3) = 10.0;
  x_ref(0) = 10.0;
  x_ref(1) = 13.0;
  return ((x - x_ref).transpose() * (x - x_ref) + u.transpose() * u).value();
}

autodiff::dual2nd cost_function_diff(VectorXdual2nd x, VectorXdual2nd u)
{
  VectorXdual2nd x_ref = VectorXdual2nd::Zero(x.size());
  x_ref(3) = 10.0;
  x_ref(0) = 10.0;
  x_ref(1) = 13.0;
  return ((x - x_ref).transpose() * (x - x_ref) + u.transpose() * u).value();
}

} // namespace

iLQR_MPC::iLQR_MPC(const int horizon, const int iters) : horizon_(horizon), iters_(iters)
{
  initialize_matrices(horizon);
  std::cerr << "constructor" << std::endl;
}

// TODO: make it a free function
void iLQR_MPC::rollout(Trajectory &trajectory)
{
  double current_time_s{0.0f};
  for (size_t i = 0; i + 1 < static_cast<size_t>(horizon_); ++i)
  {
    trajectory.times.at(i) = current_time_s;
    trajectory.actions(0, i) += 0.1;
    trajectory.actions(1, i) += 0.1;
    DynamicsT::step(trajectory.states.col(i), trajectory.actions.col(i), trajectory.states.col(i + 1));
    current_time_s += DynamicsT::ts;
  }
  trajectory.times.at(horizon_ - 1) = current_time_s;

  // cout << "trajectory.states: " << trajectory.states << endl;
  // cout << "trajectory.actions: " << trajectory.actions << endl;
}

void iLQR_MPC::solve(const Ref<State> &x0)
{
  auto trajectory = Trajectory(horizon_);
  this->rollout(trajectory);

  for (int i{0}; i < iters_; ++i)
  {
    this->backward_pass(trajectory);
    auto J = compute_cost(trajectory);
    cout << "iter: " << i << " cost: " << J << endl;

    for (int j{1}; j < 10; ++j)
    {
      auto new_trajectory = this->forward_pass(trajectory, /* alpha= */ pow(0.5, i));
      auto J_new = compute_cost(new_trajectory);
      // cout << "J new: " << J_new << endl;
      if (J_new < J)
      {
        trajectory = new_trajectory;
        J = J_new;
        break;
      }
    }
  }

  std::cerr << "rollout" << std::endl;
}

iLQR_MPC::Trajectory iLQR_MPC::forward_pass(const Trajectory &trajectory, const double alpha)
{
  Trajectory new_trajectory = Trajectory(horizon_);

  new_trajectory.states.col(0) = trajectory.states.col(0);
  for (size_t i = 0; i + 1 < static_cast<size_t>(horizon_); ++i)
  {
    // u[t] = u_bar[t] + alpga * k[t] + K[t] * (x[t] - x_bar[t])
    new_trajectory.actions.col(i) =
        trajectory.actions.col(i) + alpha * k[i] + K[i] * (new_trajectory.states.col(i) - trajectory.states.col(i));

    DynamicsT::step(new_trajectory.states.col(i), new_trajectory.actions.col(i), new_trajectory.states.col(i + 1));
  }

  return new_trajectory;
}

double iLQR_MPC::compute_cost(const Trajectory &trajectory)
{
  double cost = 0.0;
  for (int i{0}; i + 1 < horizon_; ++i)
  {
    cost += cost_function(trajectory.states.col(i), trajectory.actions.col(i));
  }
  return cost;
}

void iLQR_MPC::backward_pass(const Trajectory &trajectory)
{
  this->compute_matrices(trajectory);

  Vector Vx = lx.back();
  Matrix Vxx = lxx.back();

  constexpr bool debug = false;
  if (debug)
  {
    cout << "Vx " << endl << Vx << endl;
    cout << "Vxx " << endl << Vxx << endl;
  }

  Matrix I = Matrix::Identity(DynamicsT::state_size, DynamicsT::state_size);

  for (int i = horizon_ - 1; i >= 0; i--)
  {
    Vector Qx = lx[i] + fx[i].transpose() * Vx;
    Vector Qu = lu[i] + fu[i].transpose() * Vx;

    // cout << "Qx[i] = " << endl << Qx << endl;
    // cout << "Qu[i] = " << endl << Qu << endl;

    Matrix Qxx = lxx[i] + fx[i].transpose() * Vxx * fx[i];
    Matrix Quu = luu[i] + fu[i].transpose() * (Vxx + (mu_ * I)) * fu[i];
    Matrix Qux = lux[i] + fu[i].transpose() * (Vxx + (mu_ * I)) * fx[i];

    for (int j{0}; j < K[i].cols(); ++j)
    {
      K[i].col(j) = -Quu.ldlt().solve(Qux.col(j));
    }
    k[i] = -Quu.ldlt().solve(Qu);

    // cout << "Quu = " << endl << Quu_inv << endl;
    // cout << "Quu_inv = " << endl << Quu_inv << endl;
    // cout << "Qux = " << endl << Qux << endl;
    // cout << "Qu = " << endl << Qu << endl;
    // cout << "K[i] = " << endl << K[i] << endl;
    // cout << "k[i] = " << endl << k[i] << endl;

    Vector Vx = Qx + K[i].transpose() * Quu * k[i] + K[i].transpose() * Qu + Qux.transpose() * k[i];
    Matrix Vxx = Qxx + K[i].transpose() * Quu * K[i] + K[i].transpose() * Qux + Qux.transpose() * K[i];

    Vxx = 0.5 * (Vxx + Vxx.transpose());
  }
}

void iLQR_MPC::compute_matrices(const Trajectory &trajectory)
{
  for (int i = 0; i < horizon_; ++i)
  {
    fx[i] = calc_jacobian_x(DynamicsT::step_, trajectory.states.col(i), trajectory.actions.col(i));
    fu[i] = calc_jacobian_u(DynamicsT::step_, trajectory.states.col(i), trajectory.actions.col(i));
    lx[i] = calc_grad_x(cost_function, trajectory.states.col(i), trajectory.actions.col(i));
    lu[i] = calc_grad_u(cost_function, trajectory.states.col(i), trajectory.actions.col(i));

    lxx[i] = calc_hessian_x(cost_function, trajectory.states.col(i), trajectory.actions.col(i), 0.01);
    luu[i] = calc_hessian_u(cost_function, trajectory.states.col(i), trajectory.actions.col(i), 0.01);
    lux[i] = calc_hessian_ux(cost_function, trajectory.states.col(i), trajectory.actions.col(i), 0.01);
  }

  lx[horizon_] = calc_grad_x(cost_function, trajectory.states.col(horizon_ - 1), trajectory.actions.col(horizon_ - 1));
  lxx[horizon_] =
      calc_hessian_x(cost_function, trajectory.states.col(horizon_ - 1), trajectory.actions.col(horizon_ - 1));
}

void iLQR_MPC::initialize_matrices(const int horizon)
{
  const int state_size = DynamicsT::state_size;
  const int action_size = DynamicsT::action_size;

  for (int i = 0; i < horizon; i++)
  {
    fx.push_back(Matrix::Zero(state_size, state_size));
    fu.push_back(Matrix::Zero(action_size, action_size));
    lx.push_back(Vector::Zero(state_size));
    lxx.push_back(Matrix::Zero(state_size, state_size));
    lu.push_back(Action::Zero(action_size));
    luu.push_back(Matrix::Zero(action_size, action_size));
    lux.push_back(Matrix::Zero(action_size, state_size));

    K.push_back(Matrix::Zero(action_size, state_size));
    k.push_back(Vector::Zero(action_size));
  }

  // append derivatives of terminal state
  lx.push_back(Vector::Zero(state_size));
  lxx.push_back(Matrix::Zero(state_size, state_size));
}

} // namespace RoboticsSandbox
