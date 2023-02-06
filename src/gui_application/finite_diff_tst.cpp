#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <iostream>

#include "common/finite_diff.hpp"
#include "common/types.hpp"

namespace mpex {

using namespace Eigen;
using namespace std;
using namespace autodiff;

Vector3d dynamics(Vector3d state, Vector3d input)
{
    return 5 * state.array().square() + input(0) + input(2);
};

double cost_function(Vector x, Vector u)
{
    Vector x_ref = Vector::Zero(x.size());
    x_ref(2) = u(1);
    return ((x - x_ref).transpose() * (x - x_ref) + u.transpose() * u)(0);
}

autodiff::real cost_function_r(const VectorXreal &x, const VectorXreal &u)
{
    VectorXreal x_ref = VectorXreal::Zero(x.size());
    x_ref(2) = u(1);
    return ((x - x_ref).transpose() * (x - x_ref) + u.transpose() * u).value();
}

autodiff::dual2nd cost_function_rd(const VectorXdual2nd &x, const VectorXdual2nd &u)
{
    VectorXdual2nd x_ref = VectorXdual2nd::Zero(x.size());
    x_ref(2) = u(1);
    return ((x - x_ref).transpose() * (x - x_ref) + u.transpose() * u).value();
}

autodiff::dual2nd cost_function_full(const VectorXdual2nd &x)
{
    VectorXdual2nd x_ref = VectorXdual2nd::Zero(x.size());
    x_ref(2) = x(7);
    return ((x - x_ref).transpose() * (x - x_ref)).value();
}

TEST_CASE("Test finite differences")
{
    Vector state;
    state << 1.0, 2.0, 3.0;

    Matrix jacobian_x = calc_jacobian_x(dynamics, state, state, 0.01);
    Matrix expected_x = Matrix::Zero(3, 3);
    expected_x << 10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 30.0;
    REQUIRE(expected_x.isApprox(jacobian_x));

    Matrix jacobian_u = calc_jacobian_u(dynamics, state, state, 0.00001);
    Matrix expected_u = Matrix::Zero(3, 3);
    expected_u << 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0;
    REQUIRE(expected_u.isApprox(jacobian_u));
    REQUIRE(expected_u == jacobian_u);
}

TEST_CASE("Test finite differences bycicle model")
{

    EigenKinematicBicycle::State state;
    state << 0.0, 0.0, 1.0, 1.0, 0.5, 0.5;
    EigenKinematicBicycle::Action action;
    action << 0.1, 0.1;

    VectorXreal state_r(state.size());
    state_r << 0.0, 0.0, 1.0, 1.0, 0.5, 0.5;
    VectorXreal action_r(action.size());
    action_r << 0.1, 0.1;

    VectorXdual state_d(state.size());
    state_d << 0.0, 0.0, 1.0, 1.0, 0.5, 0.5;
    VectorXdual action_d(action.size());
    action_d << 0.1, 0.1;

    VectorXdual2nd state_rd(state.size());
    state_rd << 0.0, 0.0, 1.0, 1.0, 0.5, 0.5;
    VectorXdual2nd action_rd(action.size());
    action_rd << 0.1, 0.1;

    const auto new_state = EigenKinematicBicycle::step_(state, action);

    Matrix Jx = calc_jacobian_x(EigenKinematicBicycle::step_, state, action, 0.01);
    Matrix Ju = calc_jacobian_u(EigenKinematicBicycle::step_, state, action, 0.01);
    cout << "Jx " << endl << Jx << endl;
    cout << "Ju " << endl << Ju << endl;

    Matrix Jx_ = jacobian(EigenKinematicBicycle::step_diff<VectorXdual>, wrt(state_d), at(state_d, action_d));
    Matrix Ju_ = jacobian(EigenKinematicBicycle::step_diff<VectorXdual>, wrt(action_d), at(state_d, action_d));
    cout << "Jx_ " << endl << Jx_ << endl;
    cout << "Ju_ " << endl << Ju_ << endl;

    Vector Gx = calc_grad_x(cost_function, state, action);
    Vector Gu = calc_grad_u(cost_function, state, action);
    cout << "Gx " << Gx << endl;
    cout << "Gu " << Gu << endl;

    Vector g_x = gradient(cost_function_r, wrt(state_r), at(state_r, action_r));
    Vector g_u = gradient(cost_function_r, wrt(action_r), at(state_r, action_r));

    cout << "g_x= " << g_x.transpose() << endl;
    cout << "g_u= " << g_u.transpose() << endl;

    dual2nd ux, uu; // the output scalar u = f(x) evaluated together with Hessian below
    VectorXdual gx, gu;

    Matrix H_x = hessian(cost_function_rd, wrt(state_rd), at(state_rd, action_rd), ux, gx);
    Matrix H_u = hessian(cost_function_rd, wrt(action_rd), at(state_rd, action_rd), uu, gu);
    cout << "H_x= " << H_x << endl;
    cout << "H_u= " << H_u << endl;
    cout << "g_x= " << gx.transpose() << endl;
    cout << "g_u= " << gu.transpose() << endl;

    VectorXdual2nd state_full(state.size() + action.size());
    state_full << 0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.1, 0.1;
    Matrix H_full = hessian(cost_function_full, wrt(state_full), at(state_full));
    cout << "H_full= " << H_full << endl;

    Matrix H_x_ = calc_hessian_x(cost_function, state, action, 0.01);
    Matrix H_u_ = calc_hessian_u(cost_function, state, action, 0.01);
    Matrix H_ux_ = calc_hessian_ux(cost_function, state, action, 0.01);
    cout << "H_x_= " << H_x_ << endl;
    cout << "H_u_= " << H_u_ << endl;
    cout << "H_ux_= " << H_ux_ << endl;
}

} // namespace mpex
