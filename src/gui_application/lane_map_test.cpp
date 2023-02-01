#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

#include "environment/lane_map.hpp"

namespace mpex
{

namespace environment
{

TEST_CASE("Test pnpoly")
{
  // define unit box
  constexpr int nvert{4};
  double *x_vals = new double[nvert]{0, 1, 1, 0};
  double *y_vals = new double[nvert]{1, 1, 0, 0};

  SECTION("Test inside of box")
  {
    const double x = 0.5;
    const double y = 0.5;
    CHECK(pnpoly(nvert, x_vals, y_vals, x, y));
  }

  SECTION("Test outside of box")
  {
    const double x = 1.5;
    const double y = 1.5;
    CHECK_FALSE(pnpoly(nvert, x_vals, y_vals, x, y));
  }
}

TEST_CASE("Test is_between_polylines")
{
  using geometry::Point2D;
  using geometry::Polyline2D;

  // define unit box
  constexpr int nvert{4};
  const auto polyline_a = Polyline2D({0, 1}, {1, 1});
  const auto polyline_b = Polyline2D({0, 1}, {0, 0});

  SECTION("Test inside of box")
  {
    const double x = 0.5;
    const double y = 0.5;
    CHECK(is_between_polylines(polyline_a, polyline_b, Point2D(x, y)));
  }

  SECTION("Test outside of box")
  {
    const double x = 1.5;
    const double y = 1.5;
    CHECK_FALSE(is_between_polylines(polyline_a, polyline_b, Point2D(x, y)));
  }
}
} // namespace environment

} // namespace mpex
