// Copyright © 2022 <copyright holders>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the “Software”), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE

#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <gui_application/environment/lane_map.hpp>
#include <gui_application/geometry/geometry.hpp>
#include <gui_application/third_party/json/single_include/nlohmann/json.hpp>

namespace mpex { namespace environment {

class Track
{

    using Spline = geometry::AlglibCubic2DSpline;

    static constexpr int kSubsampleRes = 1;
    static constexpr double kTrackScaling = 50.0;

    void create_corridor_from_data()
    {
        geometry::Polyline2D left_boundary(data_["X_i"], data_["Y_i"]);
        geometry::Polyline2D right_boundary(data_["X_o"], data_["Y_o"]);
        geometry::Polyline2D centerline = environment::Track::compute_centerline(left_boundary, right_boundary);
        corridor_ = environment::Corridor(centerline, left_boundary, right_boundary);
    }

    static std::shared_ptr<geometry::AlglibCubic2DSpline> create_spline_from_polyline(const geometry::Polyline2D &polyline)
    {
        const auto &arclengths_m = polyline.get_arclength();
        auto x_values = std::vector<double>(arclengths_m.size(), 0);
        auto y_values = std::vector<double>(arclengths_m.size(), 0);
        Eigen::VectorXd::Map(&x_values[0], x_values.size()) = polyline.get_data().col(0);
        Eigen::VectorXd::Map(&y_values[0], y_values.size()) = polyline.get_data().col(1);
        return std::make_shared<geometry::AlglibCubic2DSpline>(arclengths_m, x_values, y_values);
    }

  public:
    Track() = default;

    Track(const std::unordered_map<std::string, std::vector<double>> &data) : data_(std::move(data))
    {
        create_corridor_from_data();
        centerline_spline_ptr_ = Track::create_spline_from_polyline(corridor_.get_centerline());
    };

    Track(const std::string &filename) : Track(Track::read_track_from_json(filename)){};

    static std::unordered_map<std::string, std::vector<double>> read_track_from_json(const std::string &json_path)
    {
        std::ifstream file(json_path.c_str());

        nlohmann::json json;
        file >> json;

        std::unordered_map<std::string, std::vector<double>> data;
        data["X"] = static_cast<std::vector<double>>(json["X"]);
        data["Y"] = static_cast<std::vector<double>>(json["Y"]);
        data["X_i"] = static_cast<std::vector<double>>(json["X_i"]);
        data["Y_i"] = static_cast<std::vector<double>>(json["Y_i"]);
        data["X_o"] = static_cast<std::vector<double>>(json["X_o"]);
        data["Y_o"] = static_cast<std::vector<double>>(json["Y_o"]);

        return data;
    }
    static Polyline2D compute_centerline(const Polyline2D &left_boundary, const Polyline2D &right_boundary)
    {
        geometry::Polyline2D centerline;
        for (size_t i{0}; i < left_boundary.size(); ++i)
        {
            // Transform to vector because we have defined arithmetic operations
            auto p_left = geometry::Vector2D(left_boundary[i]);
            auto p_right = geometry::Vector2D(right_boundary[i]);
            auto p_center = (p_right + p_left) / 2.0;
            centerline.push_back(p_center.get_point());
        }

        return centerline;
    }

    [[nodiscard]] Track scale(const double scale) const
    {
        decltype(data_) scaled_data = data_;
        for (auto &[key, values] : scaled_data)
        {
            std::transform(values.begin(), values.end(), values.begin(), [&](auto v) { return scale * v; });
        }
        return Track(scaled_data);
    }

    [[nodiscard]] Track calc_subsampled_track() const
    {
        std::unordered_map<std::string, std::vector<double>> subsampled_data;
        for (const auto &[key, values] : data_)
        {
            for (size_t i{0}; i < values.size(); i += kSubsampleRes)
            {
                subsampled_data[key].push_back(values[i]);
            }
        }
        return Track(subsampled_data);
    }

    [[nodiscard]] double eval_curvature(const double s_unclamped) const
    {
        const auto &arclength = corridor_.get_centerline().get_arclength();
        const auto s = std::fmod(s_unclamped, arclength.back());
        return centerline_spline_ptr_->eval_curvature(s);
    }

    [[nodiscard]] SE2 convert_to_se2(const double s_unclamped, const double d, const double mu) const
    {
        const auto &arclength = corridor_.get_centerline().get_arclength();
        const auto s = std::fmod(s_unclamped, arclength.back());

        const auto &x_spline = centerline_spline_ptr_->x_spline_;
        const auto &y_spline = centerline_spline_ptr_->y_spline_;

        double x, dx, ddx;
        double y, dy, ddy;

        x_spline->eval_diff(s, x, dx, ddx);
        y_spline->eval_diff(s, y, dy, ddy);

        auto normal = geometry::Vector2D(geometry::Point2D{-dy, dx}).normalized();
        double new_x = x + d * normal[0];
        double new_y = y + d * normal[1];

        double theta = std::atan2(dy, dx) + mu;
        return SE2{new_x, new_y, theta};
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////// Getters
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    [[nodiscard]] std::unordered_map<std::string, std::vector<double>> &get_data_mutable()
    {
        return data_;
    }

    [[nodiscard]] const Corridor &get_corridor() const
    {
        return corridor_;
    }

  private:
    std::unordered_map<std::string, std::vector<double>> data_;
    std::shared_ptr<Spline> centerline_spline_ptr_;
    Corridor corridor_;
};

}} // namespace mpex::environment