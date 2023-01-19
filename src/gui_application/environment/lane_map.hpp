#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace mpex
{

namespace geometry
{

class Point2D
{

public:
  Point2D() = default;

  Point2D(const Eigen::Vector2d &data) : data_(data){};

  Point2D(const double x, const double y)
  {
    data_ = Eigen::Vector2d::Zero();
    data_ << x, y;
  };

  [[nodiscard]] inline double x() const noexcept
  {
    return data_(0);
  };

  [[nodiscard]] inline double y() const noexcept
  {
    return data_(1);
  };

private:
  Eigen::Vector2d data_;
};

class Polyline2D
{
public:
  Polyline2D() = default;

  Polyline2D(Eigen::Matrix2d &data) : data_(data){};

  Polyline2D(const std::vector<double> &x, const std::vector<double> &y)
  {
    assert(x.size() == y.size());
    data_ = Eigen::MatrixXd::Zero(2, x.size());
    for (int i{0}; i < x.size(); ++i)
    {
      data_(0, i) = x[i];
      data_(1, i) = y[i];
    }
  };

  [[nodiscard]] Point2D get_point(std::size_t i) const
  {
    return Point2D(data_.col(i));
  };

  [[nodiscard]] std::size_t size() const
  {
    return data_.cols();
  };

  Eigen::Matrix2Xd get_data() const
  {
    return data_;
  }

private:
  Eigen::Matrix2Xd data_;
};

} // namespace geometry

namespace environment
{

using geometry::Point2D;
using geometry::Polyline2D;

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
}

[[nodiscard]] bool is_between_polylines(const Polyline2D &left_boundary, const Polyline2D &right_boundary, const Point2D point)
{
  const int nvert = left_boundary.size() + right_boundary.size();

  Eigen::VectorXd x_vals;
  Eigen::VectorXd y_vals;
  x_vals.resize(nvert);
  y_vals.resize(nvert);

  x_vals.head(left_boundary.size()) = left_boundary.get_data().row(0);
  x_vals.tail(right_boundary.size()) = right_boundary.get_data().rowwise().reverse().row(0);

  y_vals.head(left_boundary.size()) = left_boundary.get_data().row(1);
  y_vals.tail(right_boundary.size()) = right_boundary.get_data().rowwise().reverse().row(1);

  return pnpoly(nvert, x_vals.data(), y_vals.data(), point.x(), point.y());
};

using LaneID = uint32_t;

class Lane
{

public:
  Lane() = default;
  Lane(Polyline2D centerline, Polyline2D left_boundary, Polyline2D right_boundary)
      : centerline_(centerline), left_boundary_(left_boundary), right_boundary_(right_boundary){};

  [[nodiscard]] inline bool is_inside(const Point2D point) const noexcept
  {
    return is_between_polylines(left_boundary_, right_boundary_, point);
  };

private:
  Polyline2D centerline_;
  Polyline2D left_boundary_;
  Polyline2D right_boundary_;
};

using LaneMap = std::unordered_map<uint32_t, Lane>;
} // namespace environment

} // namespace mpex
