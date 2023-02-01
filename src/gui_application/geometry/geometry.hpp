#pragma once

#include <Magnum/Magnum.h>
#include <limits>
#include <vector>

#include <Eigen/Dense>

namespace geometry
{

using namespace Eigen;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////// Point2D Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

  [[nodiscard]] double x() const noexcept
  {
    return data_(0);
  };

  [[nodiscard]] double y() const noexcept
  {
    return data_(1);
  };

private:
  Eigen::Vector2d data_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////// Polyline2D Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Polyline2D
{

public:
  Polyline2D() = default;
  Polyline2D(Eigen::Matrix2d &data);
  Polyline2D(const std::vector<double> &x, const std::vector<double> &y);
  [[nodiscard]] Point2D get_point(std::size_t i) const;
  Point2D operator[](std::size_t i) const;
  void push_back(const Point2D &point);
  void recompute_arclength();
  [[nodiscard]] std::size_t size() const;
  Eigen::MatrixX2d get_data() const;
  double calc_curvilinear_coord(const Vector2d &point, double *signed_distance_m = nullptr) const;

private:
  Eigen::MatrixX2d data_;
  std::vector<double> arclength_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////// Vector2D Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Vector2D : public Eigen::Vector2d
{
public:
  Vector2D() = default;

  Vector2D(const Eigen::Vector2d &v)
  {
    Eigen::Vector2d::operator=(v);
  };
  Vector2D(const Point2D &p)
  {
    (*this)[0] = p.x();
    (*this)[1] = p.y();
  };

  Vector2D(const Point2D &p1, const Point2D &p2)
  {
    (*this)[0] = p2.x() - p1.x();
    (*this)[1] = p2.y() - p1.y();
  };

  Point2D get_point()
  {
    return Point2D((*this)[0], (*this)[1]);
  }

  Vector2D operator+(const Vector2D &v)
  {
    return static_cast<Vector2D>(Eigen::Vector2d::operator+(v));
  };

  Vector2D operator-(const Vector2D &v)
  {
    return static_cast<Vector2D>(Eigen::Vector2d::operator-(v));
  };
  Vector2D operator*(const double v)
  {
    return static_cast<Vector2D>(Eigen::Vector2d::operator*(v));
  };
  Vector2D operator/(const double v)
  {
    return static_cast<Vector2D>(Eigen::Vector2d::operator/(v));
  };
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////// Polyline2D Utilities
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::size_t calc_closest_point(const Polyline2D &polyline, const Point2D &point);
std::size_t calc_closest_point(const Polyline2D &polyline, const Vector2d &point);
std::size_t calc_closest_point_in_interval(const Polyline2D &polyline, const Point2D &point, std::size_t x_min, std::size_t x_max);
double calc_curvilinear_coord(const Polyline2D &polyline, const std::vector<double> &knots, const Point2D &p, double *signed_distance_m);

} // namespace geometry