#include <Eigen/Dense>
#include <iostream>

#include "types.hpp"

namespace mpex {

using namespace std;

template <typename Function, typename State, typename Action>
Matrix calc_jacobian_x(Function dynamics, const State &x, const Action &u, double dt = 0.001)
{
    Matrix Jx = Matrix::Zero(x.size(), x.size());
    for (int j{0}; j < x.size(); ++j)
    {
        Vector dx = Vector::Zero(x.size());
        dx(j) = dt;
        Jx.col(j) = 0.5 * (dynamics(x + dx, u) - dynamics(x - dx, u)) / dt;
    }
    return Jx;
}

template <typename Function, typename State, typename Action>
Matrix calc_hessian_x(Function f, const State &x, const Action &u, double dt = 0.001)
{
    Matrix H = Matrix::Zero(x.size(), x.size());
    for (int i{0}; i < x.size(); ++i)
    {
        for (int j{i}; j < x.size(); ++j)
        {
            Vector dx_i = Vector::Zero(x.size());
            Vector dx_j = Vector::Zero(x.size());
            dx_i(i) = dt;
            dx_j(j) = dt;
            H(i, j) = (f(x + dx_i + dx_j, u) - f(x + dx_i - dx_j, u) - f(x - dx_i + dx_j, u) + f(x - dx_i - dx_j, u)) / (4.0 * dt * dt);
        }
    }
    return H;
}

template <typename Function, typename State, typename Action>
Matrix calc_hessian_u(Function f, const State &x, const Action &u, double dt = 0.001)
{
    Matrix H = Matrix::Zero(u.size(), u.size());
    for (int i{0}; i < u.size(); ++i)
    {
        for (int j{i}; j < u.size(); ++j)
        {
            Vector du_i = Vector::Zero(u.size());
            Vector du_j = Vector::Zero(u.size());
            du_i(i) = dt;
            du_j(j) = dt;
            H(i, j) = H(j, i) =
                (f(x, u + du_i + du_j) - f(x, u + du_i - du_j) - f(x, u - du_i + du_j) + f(x, u - du_i - du_j)) / (4.0 * dt * dt);
        }
    }
    return H;
}

template <typename Function, typename State, typename Action>
Matrix calc_hessian_ux(Function f, const State &x, const Action &u, double dt = 0.001)
{
    Matrix H = Matrix::Zero(u.size(), x.size());
    for (int i{0}; i < u.size(); ++i)
    {
        for (int j{0}; j < x.size(); ++j)
        {
            Vector du_i = Vector::Zero(u.size());
            Vector dx_j = Vector::Zero(x.size());
            du_i(i) = dt;
            dx_j(j) = dt;
            H(i, j) = (f(x + dx_j, u + du_i) - f(x - dx_j, u + du_i) - f(x + dx_j, u - du_i) + f(x - dx_j, u - du_i)) / (4.0 * dt * dt);
        }
    }
    return H;
}

template <typename Function, typename State, typename Action>
Matrix calc_grad_x(Function function, const State &x, const Action &u, double dt = 0.001)
{
    Vector grad = Vector::Zero(x.size());
    for (int j{0}; j < x.size(); ++j)
    {
        Vector dx = Vector::Zero(x.size());
        dx(j) = dt;
        grad(j) = 0.5 * (function(x + dx, u) - function(x - dx, u)) / dt;
    }
    return grad;
}

template <typename Function, typename State, typename Action>
Matrix calc_grad_u(Function function, const State &x, const Action &u, double dt = 0.001)
{
    Vector grad = Vector::Zero(u.size());
    for (int j{0}; j < u.size(); ++j)
    {
        Vector du = Vector::Zero(u.size());
        du(j) = dt;
        grad(j) = 0.5 * (function(x, u + du) - function(x, u - du)) / dt;
    }
    return grad;
}

template <typename Function, typename State, typename Action>
Matrix calc_jacobian_u(Function dynamics, const State &x, const Action &u, double dt = 0.001)
{
    Matrix Ju = Matrix::Zero(x.size(), u.size());
    for (int j{0}; j < u.size(); ++j)
    {
        Vector du = Vector::Zero(u.size());
        du(j) = dt;
        Ju.col(j) = 0.5 * (dynamics(x, u + du) - dynamics(x, u - du)) / dt;
    }
    return Ju;
}

} // namespace mpex