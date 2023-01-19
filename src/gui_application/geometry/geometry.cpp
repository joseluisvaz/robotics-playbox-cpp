#include <Eigen/Dense>
#include <cstddef>
#include <easy/profiler.h>
#include <iostream>

#include "geometry.hpp"
#include "geometry/geometry.hpp"

namespace geometry
{

P2D::P2D(const Eigen::Vector2d &data) : data_(data){};

P2D::P2D(const double x, const double y)
{
  data_ = Eigen::Vector2d::Zero();
  data_ << x, y;
};

[[nodiscard]] double P2D::x() const noexcept
{
  return data_(0);
};

[[nodiscard]] double P2D::y() const noexcept
{
  return data_(1);
};

Polyline2D::Polyline2D(Eigen::Matrix2d &data) : data_(data){};

Polyline2D::Polyline2D(const std::vector<double> &x, const std::vector<double> &y)
{
  assert(x.size() == y.size());
  data_ = Eigen::MatrixXd::Zero(2, x.size());
  for (size_t i{0}; i < x.size(); ++i)
  {
    data_(0, i) = x[i];
    data_(1, i) = y[i];
  }

  // Needs to be monotonically increasing ...
  arclength_ = std::vector<double>(data_.cols(), 0.0);
  for (size_t i{1}; i < x.size(); ++i)
  {
    // vector from p1 to p2
    arclength_[i] = (data_.col(i) - data_.col(i - 1)).norm() + arclength_[i - 1];
    assert(arclength_[i] > arclength_[i - 1]);
  }
};

[[nodiscard]] P2D Polyline2D::get_point(std::size_t i) const
{
  return P2D(data_.col(i));
};

[[nodiscard]] std::size_t Polyline2D::size() const
{
  return data_.cols();
};

Eigen::Matrix2Xd Polyline2D::get_data() const
{
  return data_;
}

double Polyline2D::calc_progress_coord(const P2D &p) const
{
  EASY_FUNCTION(profiler::colors::Yellow);
  const auto calc_projection_at_index = [&](std::size_t curr_index)
  {
    const auto a = get_point(curr_index);
    const auto b = get_point(curr_index + 1);

    auto a_to_b = geometry::V2D(a, b);
    auto a_to_p = geometry::V2D(a, p);

    return static_cast<double>(a_to_p.dot(a_to_b));
  };

  const auto curr_index = calc_closest_point(*this, p);

  // If it is at the end then check if the projection is past the norm of the vector
  if (curr_index == this->size() - 1)
  {
    const auto projection = calc_projection_at_index(curr_index - 1ULL);

    const auto a = get_point(curr_index - 1);
    const auto b = get_point(curr_index);
    auto this_vector_length = geometry::V2D(a, b).norm();

    // Saturate the arclength we are past the end of the polyline
    return projection >= this_vector_length ? arclength_.back() : arclength_[curr_index - 1] + projection;
  }

  const auto projection = calc_projection_at_index(curr_index);
  if (projection >= 0.0)
  {
    return arclength_[curr_index] + projection;
  }
  else
  {
    return curr_index >= 1 ? arclength_[curr_index - 1ULL] + calc_projection_at_index(curr_index - 1ULL) : 0.0;
  }

  EASY_END_BLOCK;
  return 0.0;
}

std::size_t calc_closest_point(const Polyline2D &polyline, const P2D &point)
{
  return calc_closest_point_in_interval(polyline, point, 0, polyline.size());
}

std::size_t calc_closest_point_in_interval(const Polyline2D &polyline, const P2D &point, std::size_t x_min, std::size_t x_max)
{
  std::size_t argmin = -1; // bug here
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

} // namespace geometry