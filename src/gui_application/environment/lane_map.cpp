#include "geometry/geometry.hpp"

#include "environment/lane_map.hpp"
namespace mpex
{
namespace environment
{

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

[[nodiscard]] bool is_between_polylines(const Polyline2D &left_boundary, const Polyline2D &right_boundary, const P2D point)
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

Lane::Lane(Polyline2D centerline, Polyline2D left_boundary, Polyline2D right_boundary)
    : centerline_(centerline), left_boundary_(left_boundary), right_boundary_(right_boundary){};

[[nodiscard]]  bool Lane::is_inside(const P2D point) const noexcept
{
  return is_between_polylines(left_boundary_, right_boundary_, point);
};

} // namespace environment
} // namespace mpex