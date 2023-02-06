#pragma once
#include <Eigen/Dense>
#include <random>

namespace mpex {

using namespace Eigen;

double angle_diff(double a, double b);

#ifndef PI_D
#define PI_D (3.14159265358979323846)
#endif

#ifndef RAD2DEG
#define RAD2DEG(a) (a * 180.0 / PI_D)
#endif

#ifndef DEG2RAD
#define DEG2RAD(a) (a * PI_D / 180.0)
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
        static std::normal_distribution<double> dist;

        return mean_.array() +
               stddev_.array() *
                   (MatrixT::Zero(mean_.rows(), mean_.cols()).unaryExpr([&](auto /* unused */) { return dist(gen); }).array());
    }

    MatrixT mean_;
    MatrixT stddev_;
};

} // namespace mpex