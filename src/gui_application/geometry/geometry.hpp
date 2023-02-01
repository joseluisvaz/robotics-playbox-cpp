#pragma once

#include <Magnum/Magnum.h>
#include <limits>
#include <vector>

#include <Eigen/Dense>

namespace geometry
{

class P2D
{

public:
  P2D() = default;
  P2D(const Eigen::Vector2d &data);
  P2D(const double x, const double y);
  [[nodiscard]] double x() const noexcept;
  [[nodiscard]] double y() const noexcept;

private:
  Eigen::Vector2d data_;
};

class Polyline2D
{

public:
  Polyline2D() = default;
  Polyline2D(Eigen::Matrix2d &data);
  Polyline2D(const std::vector<double> &x, const std::vector<double> &y);
  [[nodiscard]] P2D get_point(std::size_t i) const;
  P2D operator[](std::size_t i) const;
  void push_back(const P2D &point);
  void recompute_arclength();
  [[nodiscard]] std::size_t size() const;
  Eigen::MatrixX2d get_data() const;
  double calc_progress_coord(const P2D &point) const;

private:
  Eigen::MatrixX2d data_;
  std::vector<double> arclength_;
};

class V2D : public Eigen::Vector2d
{
public:
  V2D() = default;

  V2D(const Eigen::Vector2d &v)
  {
    Eigen::Vector2d::operator=(v);
  };
  V2D(const P2D &p)
  {
    (*this)[0] = p.x();
    (*this)[1] = p.y();
  };

  V2D(const P2D &p1, const P2D &p2)
  {
    (*this)[0] = p2.x() - p1.x();
    (*this)[1] = p2.y() - p1.y();
  };

  P2D get_point()
  {
    return P2D((*this)[0], (*this)[1]);
  }

  V2D operator+(const V2D &v)
  {
    return static_cast<V2D>(Eigen::Vector2d::operator+(v));
  };

  V2D operator-(const V2D &v)
  {
    return static_cast<V2D>(Eigen::Vector2d::operator-(v));
  };
  V2D operator*(const double v)
  {
    return static_cast<V2D>(Eigen::Vector2d::operator*(v));
  };
  V2D operator/(const double v)
  {
    return static_cast<V2D>(Eigen::Vector2d::operator/(v));
  };
};

std::size_t calc_closest_point(const Polyline2D &polyline, const P2D &point);
std::size_t calc_closest_point_in_interval(const Polyline2D &polyline, const P2D &point, std::size_t x_min, std::size_t x_max);

} // namespace geometry