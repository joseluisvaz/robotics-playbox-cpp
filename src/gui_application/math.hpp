#pragma once
#include <Eigen/Dense>
#include <random>

namespace RoboticsSandbox 
{

using namespace Eigen;

float angle_diff(float a, float b);

#ifndef PI_F
#define PI_F (3.14159265358979323846f)
#endif

#ifndef RAD2DEG
#define RAD2DEG(a) (a * 180.0f / PI_F)
#endif

#ifndef DEG2RAD
#define DEG2RAD(a) (a * PI_F / 180.0f)
#endif

#ifndef ANGLE_DIFF
#define ANGLE_DIFF(a, b) (DEG2RAD(angle_diff(RAD2DEG(a), RAD2DEG(b))))
#endif

template <typename MatrixT>
struct NormalRandomVariable
{
  NormalRandomVariable() = default;

  MatrixT operator()() const
  {
    static std::mt19937 gen{std::random_device{}()};
    static std::normal_distribution<float> dist;

    return mean_.array() + stddev_.array() * (MatrixT::Zero(mean_.rows(), mean_.cols())
                                                  .unaryExpr([&](auto /* unused */) { return dist(gen); })
                                                  .array());
  }

  MatrixT mean_;
  MatrixT stddev_;
};

} // namespace GuiApplication