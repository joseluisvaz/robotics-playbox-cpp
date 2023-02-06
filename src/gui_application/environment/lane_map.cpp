#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "gui_application/environment/lane_map.hpp"
#include "gui_application/geometry/geometry.hpp"

namespace mpex { namespace environment {

using geometry::Point2D;
using geometry::Polyline2D;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////// Environment Utilities
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Checks if candidate point is inside boundaries
///    Original code from: https://wrfranklin.org/Research/Short_Notes/pnpoly.html
///
[[nodiscard]] int pnpoly(const int nvert, const double *vertx, const double *verty, const double testx, const double testy)
{
    bool is_inside = false;

    int i, j;
    for (i = 0, j = nvert - 1; i < nvert; j = i++)
    {
        // shoot a horizontal ray from each vertex
        bool is_in_y_interval = (verty[i] > testy) != (verty[j] > testy);
        bool is_on_the_left = testx < (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i];
        if (is_in_y_interval && is_on_the_left)
        {
            is_inside = !is_inside;
        }
    }
    return is_inside;
};

[[nodiscard]] bool is_between_polylines(const Polyline2D &left_boundary, const Polyline2D &right_boundary, const Point2D point)
{
    const int nvert = left_boundary.size() + right_boundary.size();

    Eigen::VectorXd x_vals;
    Eigen::VectorXd y_vals;
    x_vals.resize(nvert);
    y_vals.resize(nvert);

    x_vals.head(left_boundary.size()) = left_boundary.get_data().col(0);
    x_vals.tail(right_boundary.size()) = right_boundary.get_data().rowwise().reverse().col(0);

    y_vals.head(left_boundary.size()) = left_boundary.get_data().col(1);
    y_vals.tail(right_boundary.size()) = right_boundary.get_data().rowwise().reverse().col(1);

    return pnpoly(nvert, x_vals.data(), y_vals.data(), point.x(), point.y());
};

[[nodiscard]] Corridor create_line_helper(size_t n_points)
{
    std::vector<double> x_vals;
    std::vector<double> y_vals;
    std::vector<double> z_vals;
    for (int i{0}; i < n_points; ++i)
    {
        x_vals.push_back(i * 2.0);
        y_vals.push_back(0.0);
        z_vals.push_back(0.0);
    }

    auto centerline = Polyline2D(x_vals, y_vals);

    x_vals.clear();
    y_vals.clear();
    for (int i{0}; i < n_points; ++i)
    {
        x_vals.push_back(i * 2.0);
        y_vals.push_back(4.0);
    }
    auto left_boundary = Polyline2D(x_vals, y_vals);

    x_vals.clear();
    y_vals.clear();
    for (int i{0}; i < n_points; ++i)
    {
        x_vals.push_back(i * 2.0);
        y_vals.push_back(-4.0);
    }
    auto right_boundary = Polyline2D(x_vals, y_vals);

    return environment::Corridor(centerline, left_boundary, right_boundary);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////// Corridor Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Corridor::Corridor(Polyline2D centerline, Polyline2D left_boundary, Polyline2D right_boundary)
    : centerline_(centerline), left_boundary_(left_boundary), right_boundary_(right_boundary){};

[[nodiscard]] bool Corridor::is_inside(const Point2D point) const noexcept
{
    return is_between_polylines(left_boundary_, right_boundary_, point);
};

}} // namespace mpex::environment
