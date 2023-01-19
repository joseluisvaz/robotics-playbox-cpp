#pragma once

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
  [[nodiscard]] std::size_t size() const;
  Eigen::Matrix2Xd get_data() const;

private:
  Eigen::Matrix2Xd data_;
  std::vector<double> arclength_;
};

} // namespace geometry