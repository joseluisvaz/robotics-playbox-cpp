#include "math.hpp"

namespace mpex
{

double angle_diff(double a, double b)
{
  double dif = fmod(b - a + 180.0f, 360.0f);
  if (dif < 0)
  {
    dif += 360.0;
  }
  return dif - 180.0;
}

} // namespace mpex