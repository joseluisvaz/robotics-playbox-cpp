#include "math.hpp"

namespace GuiApplication
{

float angle_diff(float a, float b)
{
  float dif = fmod(b - a + 180.0f, 360.0f);
  if (dif < 0)
  {
    dif += 360.0f;
  }
  return dif - 180.0f;
}

} // namespace GuiApplication