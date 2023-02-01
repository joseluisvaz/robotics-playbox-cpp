#include <Eigen/Dense>
#include <cstddef>
#include <easy/profiler.h>
#include <iostream>
#include <optional>
#include <utility>

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

[[nodiscard]] P2D Polyline2D::get_point(std::size_t i) const
{
  return P2D(data_.row(i));
};

[[nodiscard]] P2D Polyline2D::operator[](std::size_t i) const
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

void Polyline2D::push_back(const P2D &point)
{
  data_.conservativeResize(data_.rows() + 1, data_.cols());
  data_(data_.rows() - 1, 0) = point.x();
  data_(data_.rows() - 1, 1) = point.y();
  recompute_arclength();
}

double Polyline2D::calc_progress_coord(const P2D &p) const
{
  EASY_FUNCTION(profiler::colors::Yellow);

  const auto calc_projection_and_vector_length = [&](const size_t curr_index)
  {
    const auto a = get_point(curr_index);
    const auto b = get_point(curr_index + 1);
    const auto a_to_b = geometry::V2D(a, b);
    const auto a_to_p = geometry::V2D(a, p);
    const double projection = a_to_p.dot(a_to_b);
    const auto this_vector_length = geometry::V2D(a, b).norm();
    return std::make_pair(projection, this_vector_length);
  };

  const auto curr_index = calc_closest_point(*this, p);

  // If it is at the end then check if the projection is past the norm of the vector
  if (curr_index == this->size() - 1) // Handle end of polyline
  {
    auto [projection, vector_length] = calc_projection_and_vector_length(curr_index - 1ULL);
    if (projection > vector_length)
    {
      return arclength_.back();
    }

    return arclength_[curr_index - 1] + projection;
  }

  if (curr_index == 0) // Handle start of polyline
  {
    auto [projection, vector_length] = calc_projection_and_vector_length(0);
    return projection < 0.0 ? arclength_.front() : arclength_[curr_index] + projection;
  }

  // Handle middle of polyline
  auto [projection, _] = calc_projection_and_vector_length(curr_index);
  if (projection < 0.0)
  {
    auto [projection_on_prev, _] = calc_projection_and_vector_length(curr_index - 1);
    return arclength_[curr_index - 1] + projection_on_prev;
  }

  return arclength_[curr_index] + projection;
}

std::size_t calc_closest_point(const Polyline2D &polyline, const P2D &point)
{
  return calc_closest_point_in_interval(polyline, point, 0, polyline.size());
}

std::size_t calc_closest_point_in_interval(const Polyline2D &polyline, const P2D &point, std::size_t x_min, std::size_t x_max)
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

} // namespace geometry