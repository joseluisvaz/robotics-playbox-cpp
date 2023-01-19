#include <Eigen/Dense>

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
  for (int i{0}; i < x.size(); ++i)
  {
    data_(0, i) = x[i];
    data_(1, i) = y[i];
  }

  // Needs to be monotonically increasing ...
  arclength_ = std::vector<double>(data_.cols(), 0.0);
  for (int i{1}; i < x.size(); ++i)
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

} // namespace geometry