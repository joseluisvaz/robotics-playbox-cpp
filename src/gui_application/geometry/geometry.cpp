#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <easy/profiler.h>
#include <iostream>
#include <optional>
#include <utility>

#include "geometry.hpp"
#include "geometry/geometry.hpp"

namespace
{

using namespace Eigen;

}
namespace geometry
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////// Polyline2D Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Polyline2D::Polyline2D(Eigen::Matrix2d &data) : data_(data){};

Polyline2D::Polyline2D(const std::vector<double> &x, const std::vector<double> &y)
{
  assert(x.size() == y.size());
  data_ = Eigen::MatrixXd::Zero(x.size(), 2);
  for (size_t i{0}; i < x.size(); ++i)
  {
    data_(i, 0) = x[i];
    data_(i, 1) = y[i];
  }

  recompute_arclength();
};

void Polyline2D::recompute_arclength()
{
  // Needs to be monotonically increasing ...
  arclength_ = std::vector<double>(data_.rows(), 0.0);
  for (size_t i{1}; i < data_.rows(); ++i)
  {
    // vector from p1 to p2
    arclength_[i] = (data_.row(i) - data_.row(i - 1)).norm() + arclength_[i - 1];
    assert(arclength_[i] > arclength_[i - 1]);
  }
};

[[nodiscard]] Point2D Polyline2D::get_point(std::size_t i) const
{
  return Point2D(data_.row(i));
};

[[nodiscard]] Point2D Polyline2D::operator[](std::size_t i) const
{
  return get_point(i);
};

[[nodiscard]] std::size_t Polyline2D::size() const
{
  return data_.rows();
};

Eigen::MatrixX2d Polyline2D::get_data() const
{
  return data_;
}

void Polyline2D::push_back(const Point2D &point)
{
  data_.conservativeResize(data_.rows() + 1, data_.cols());
  data_(data_.rows() - 1, 0) = point.x();
  data_(data_.rows() - 1, 1) = point.y();
  recompute_arclength();
}

double Polyline2D::calc_curvilinear_coord(const Vector2d &p, double *signed_distance_m) const
{
  return geometry::calc_curvilinear_coord(*this, this->arclength_, Point2D(p), signed_distance_m);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////// Polyline2D Utilities
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double calc_curvilinear_coord(const Polyline2D &polyline, const std::vector<double> &knots, const Point2D &p, double *signed_distance_m)
{
  const auto calc_projection_and_vector_length = [&](const size_t curr_index)
  {
    // Desciption of the vector operations.
    //
    //   v_ab
    //    ^
    //    |   p ^ v_ap
    //    |    /
    //    |   /
    //    |  /
    //    | /
    //    ----------------------> v_ab
    //    a                     b
    //
    //   v_ab: is the vector from point a to point b
    //   v_ap: is the vector from point b to point p
    //   ~v_ab: is the orthogonal vector from a to point b

    const auto a = polyline.get_point(curr_index);
    const auto b = polyline.get_point(curr_index + 1);
    const auto v_ab = Vector2D(a, b);
    const auto v_ap = Vector2D(a, p);
    const double s_projection = v_ap.dot(v_ab);

    auto v_ab_orth = v_ab;
    v_ab_orth(0) = -v_ab(1);
    v_ab_orth(1) = v_ab(0);

    const double d_projection = v_ap.dot(v_ab_orth);
    return std::make_pair(s_projection, d_projection);
  };

  const auto closest_point_index = calc_closest_point(polyline, p);

  // Get the appropiate vector index by clamping the value to the last element.
  auto closest_line_index = closest_point_index == polyline.size() - 1 ? closest_point_index - 1ULL : closest_point_index;
  auto [projection_tmp, _] = calc_projection_and_vector_length(closest_line_index);

  if (projection_tmp < 0) // If it projects backwards then use the previous line index, clamp it to 0
  {
    closest_line_index = closest_line_index == 0 ? 0 : closest_line_index - 1ULL;
  }

  auto [projection, distance_m] = calc_projection_and_vector_length(closest_line_index);
  if (signed_distance_m)
  {
    *signed_distance_m = distance_m;
  }
  return std::max(std::min(knots[closest_line_index] + projection, knots.back()), knots.front());
};

std::size_t calc_closest_point_in_interval(const Polyline2D &polyline, const Point2D &point, std::size_t x_min, std::size_t x_max)
{
  std::size_t argmin = 0;
  auto min_squared_dist = std::numeric_limits<double>::max();
  for (std::size_t i = x_min; i < x_max; i++)
  {
    auto this_point = polyline.get_point(i);

    auto dx = this_point.x() - point.x();
    auto dy = this_point.y() - point.y();
    auto squared_dist = dx * dx + dy * dy;
    if (squared_dist < min_squared_dist)
    {
      argmin = i;
      min_squared_dist = squared_dist;
    }
  }
  return argmin;
}

std::size_t calc_closest_point(const Polyline2D &polyline, const Point2D &point)
{
  return calc_closest_point_in_interval(polyline, point, 0, polyline.size());
}

std::size_t calc_closest_point(const Polyline2D &polyline, const Vector2d &point)
{
  return calc_closest_point_in_interval(polyline, Point2D(point), 0, polyline.size());
}

} // namespace geometry