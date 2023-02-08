#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "gui_application/geometry/geometry.hpp"

namespace mpex { namespace environment {

using geometry::Point2D;
using geometry::Polyline2D;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////// Corridor Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Corridor
{

  public:
    Corridor() = default;
    Corridor(Polyline2D centerline, Polyline2D left_boundary, Polyline2D right_boundary);
    [[nodiscard]] bool is_inside(const Point2D point) const noexcept;
    [[nodiscard]] const Polyline2D &get_centerline() const;
    [[nodiscard]] const Polyline2D &get_left_boundary() const;
    [[nodiscard]] const Polyline2D &get_right_boundary() const;

    Polyline2D centerline_;
    Polyline2D left_boundary_;
    Polyline2D right_boundary_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////// Environment Utilities
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Checks if candidate point is inside boundaries
///    Original code from: https://wrfranklin.org/Research/Short_Notes/pnpoly.html
///
[[nodiscard]] int pnpoly(const int nvert, const double *vertx, const double *verty, const double testx, const double testy);
[[nodiscard]] bool is_between_polylines(const Polyline2D &left_boundary, const Polyline2D &right_boundary, const Point2D point);
[[nodiscard]] Corridor create_line_helper(size_t n_points);

}} // namespace mpex::environment
