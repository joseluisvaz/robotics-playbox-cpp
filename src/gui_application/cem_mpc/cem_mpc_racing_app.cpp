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

#include <cstddef>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Color.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/SceneGraph.h>
#include <easy/profiler.h>

#include "base_application/base_application.hpp"
#include "cem_mpc/cem_mpc.h"
#include "cem_mpc/cem_mpc_racing_app.hpp"
#include "cem_mpc/intelligent_driver_model.hpp"
#include "common/types.hpp"
#include "environment/lane_map.hpp"
#include "environment/track_utiilities.hpp"
#include "geometry/geometry.hpp"
#include "third_party/implot/implot.h"

namespace mpex {

namespace {

constexpr size_t n_buffer_capacity = 50;

using geometry::Point2D;
using geometry::Polyline2D;

using namespace Magnum::Math::Literals;

constexpr int population = 1024;
constexpr int iters = 20;
constexpr int horizon = 20;

const auto red_color = Magnum::Math::Color3(1.0f, 0.2f, 0.0f);
const auto blue_color = Magnum::Math::Color3(0.0f, 0.6f, 1.0f);

Polyline2D compute_centerline(const Polyline2D &left_boundary, const Polyline2D &right_boundary)
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

} // namespace

CEMMPCRacingApp::CEMMPCRacingApp(const Arguments &arguments) : Magnum::Examples::BaseApplication(arguments)
{
    auto track_map_tmp = environment::read_track_from_csv("/home/vjose/code/robotics_project/src/gui_application/track.csv");

    decltype(track_map_tmp) track_map;
    for (const auto &[key, values] : track_map_tmp)
    {
        for (int i{0}; i < values.size(); i += 10)
        {
            track_map[key].push_back(values[i]);
        }
    }

    // Construct the polylines from the track information
    geometry::Polyline2D c_left_boundary(track_map["left_x"], track_map["left_y"]);
    geometry::Polyline2D c_right_boundary(track_map["right_x"], track_map["right_y"]);
    geometry::Polyline2D c_centerline = compute_centerline(c_left_boundary, c_right_boundary);

    // Map the centerline values to the x and y values that are used to fit the spline
    auto arclengths_m = c_centerline.get_arclength();
    auto x_values = std::vector<double>(arclengths_m.size(), 0);
    auto y_values = std::vector<double>(arclengths_m.size(), 0);
    Eigen::VectorXd::Map(&x_values[0], x_values.size()) = c_centerline.get_data().col(0);
    Eigen::VectorXd::Map(&y_values[0], y_values.size()) = c_centerline.get_data().col(1);
    auto cubic_spline_ptr = std::make_shared<geometry::AlglibCubic2DSpline>(arclengths_m, x_values, y_values);

    // Subsample the spline to create a more dense centerline
    Eigen::VectorXd subsampled_m = Eigen::VectorXd::LinSpaced(arclengths_m.size() * 10, arclengths_m.front(), arclengths_m.back());
    std::vector<double> x_subsampled;
    std::vector<double> y_subsampled;
    for (int i{0}; i < subsampled_m.size(); ++i)
    {
        auto point2d = cubic_spline_ptr->eval(subsampled_m(i));
        x_subsampled.push_back(point2d.x());
        y_subsampled.push_back(point2d.y());
    }
    geometry::Polyline2D densified_centerline(x_subsampled, y_subsampled);

    corridor_ = environment::Corridor(densified_centerline, c_left_boundary, c_right_boundary);

    // Create the visualization of the track
    new Graphics::LaneEntity(corridor_, scene_, vertex_color_shader_, drawable_group_);

    // Set Policy and cost function
    CEM_MPC_Config config =
        {/* iters= */ 20,
         horizon,
         /* population= */ population,
         /* elites */ 10};
    ego_policy_ptr_ = std::make_shared<CEM_MPC<EigenKinematicBicycle>>(config);

    ego_policy_ptr_->cost_function_ = std::make_shared<QuadraticCostFunction>();
    mpc_viewer_ = KinematicBicycleCemViewer(ego_policy_ptr_, scene_, wireframe_shader_, flat_shader_, drawable_group_, horizon, population);

    // Initialize current state for the simulation, kinematic bicycle.
    ego_state_ = Dynamics::State();
    ego_state_ << -10.0, 0.0, 0.0, 10.0, 0.0, 0.0;

    history_buffer_["time_s"] = containers::Buffer<double>(n_buffer_capacity);
    history_buffer_["speed_mps"] = containers::Buffer<double>(n_buffer_capacity);
    history_buffer_["accel_mpss"] = containers::Buffer<double>(n_buffer_capacity);
    history_buffer_["yaw_rad"] = containers::Buffer<double>(n_buffer_capacity);
    history_buffer_["steering_rad"] = containers::Buffer<double>(n_buffer_capacity);

    runCEM();
}

void CEMMPCRacingApp::execute()
{
    if (is_running_)
    {
        runCEM();

        camera_object_->resetTransformation()
            .translate(Magnum::Vector3::zAxis(SCALE(200.0f)))
            .translate(Magnum::Vector3::xAxis(SCALE(50.0f)))
            .rotateX(-90.0_degf)
            .rotateY(-90.0_degf)
            .translate(Magnum::Vector3::zAxis(SCALE(ego_state_[0])))
            .translate(Magnum::Vector3::xAxis(SCALE(ego_state_[1])));
    }
}

void CEMMPCRacingApp::runCEM()
{
    EASY_FUNCTION(profiler::colors::Red);

    auto trajectory = ego_policy_ptr_->solve(ego_state_, maybe_current_trajectory_);
    maybe_current_trajectory_ = trajectory;
    Dynamics::Action current_action = trajectory.actions.col(0);

    mpc_viewer_.draw();

    Vector parameters = Vector::Zero(5);
    ego_state_ = Dynamics::step_(ego_state_, current_action, parameters);
    time_s_ += Dynamics::ts;

    history_buffer_["time_s"].push(time_s_);
    history_buffer_["speed_mps"].push(ego_state_[3]);
    history_buffer_["accel_mpss"].push(ego_state_[4]);
    history_buffer_["yaw_rad"].push(ego_state_[2]);
    history_buffer_["steering_rad"].push(ego_state_[5]);

    EASY_END_BLOCK;
}

void CEMMPCRacingApp::show_menu()
{
    EASY_FUNCTION(profiler::colors::Blue);
    ImGui::SetNextWindowPos({500.0f, 50.0f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("Options", nullptr);

    if (ImGui::Button("Reset scene"))
    {
        resetCameraPosition();
        redraw();
    }

    if (ImGui::Button("Reset state"))
    {
        ego_state_ << -10.0, 0.0, 0.0, 10.0, 0.0, 0.0;
        history_buffer_["time_s"] = containers::Buffer<double>(n_buffer_capacity);
        history_buffer_["speed_mps"] = containers::Buffer<double>(n_buffer_capacity);
        history_buffer_["accel_mpss"] = containers::Buffer<double>(n_buffer_capacity);
        history_buffer_["yaw_rad"] = containers::Buffer<double>(n_buffer_capacity);
        history_buffer_["steering_rad"] = containers::Buffer<double>(n_buffer_capacity);
        redraw();
    }

    if (ImGui::Button("Run MPC"))
    {
        is_running_ = is_running_ ? false : true;
        redraw();
    }

    auto &ego_policy = *ego_policy_ptr_;

    auto &config_mutable = ego_policy.get_config_mutable();
    // ImGui::SliderInt("Num Iterations", &config_mutable.num_iters, 1, 100);
    // ImGui::SliderInt("Num Elites", &config_mutable.elites, 1, 20);
    // ImGui::SliderInt("Num Population", &config_mutable.population, 8, 2048);

    auto maybe_quadratic_cost_function_ptr = std::dynamic_pointer_cast<QuadraticCostFunction>(ego_policy.cost_function_);
    if (maybe_quadratic_cost_function_ptr)
    {
        auto &cost_function = *maybe_quadratic_cost_function_ptr;

        double v_min = 0.0;
        double v_max = 10.0;
        ImGui::SliderScalarN("Cost Weights - States", ImGuiDataType_Double, &cost_function.w_s_, 6, &v_min, &v_max, "%.3f", 0);
        ImGui::SliderScalarN("Cost Weights - Actions", ImGuiDataType_Double, &cost_function.w_a_, 2, &v_min, &v_max, "%.3f", 0);
        ImGui::SliderScalarN("Terminal Cost Weights - States", ImGuiDataType_Double, &cost_function.W_s_, 6, &v_min, &v_max, "%.3f", 0);
        ImGui::SliderScalarN("TerminalCost Weights - Actions", ImGuiDataType_Double, &cost_function.W_a_, 2, &v_min, &v_max, "%.3f", 0);

        double v_min_r = -100.0;
        double v_max_r = 100.0;
        ImGui::SliderScalarN("Ref values - States", ImGuiDataType_Double, &cost_function.r_s_, 6, &v_min_r, &v_max_r, "%.3f", 0);
        ImGui::SliderScalarN("Ref values - Actions", ImGuiDataType_Double, &cost_function.r_a_, 2, &v_min_r, &v_max_r, "%.3f", 0);
        ImGui::SliderScalarN("Terminal Ref values - States", ImGuiDataType_Double, &cost_function.R_s_, 6, &v_min_r, &v_max_r, "%.3f", 0);
        ImGui::SliderScalarN("Terminal Ref values - Actions", ImGuiDataType_Double, &cost_function.R_a_, 2, &v_min_r, &v_max_r, "%.3f", 0);
    }

    const auto make_plot = [&, this](auto title, auto value_name, auto getter_fn) {
        ImPlot::SetNextAxesToFit();
        if (ImPlot::BeginPlot(title))
        {
            ImPlot::SetupAxes("time[s]", value_name);

            // Plot history
            if (history_buffer_.find("time_s") != history_buffer_.end() && history_buffer_.find(value_name) != history_buffer_.end())
            {
                std::vector<double> t_past_values;
                std::vector<double> v_past_values;
                for (int i{0}; i < history_buffer_["time_s"].size(); ++i)
                {
                    t_past_values.push_back(history_buffer_["time_s"][i]);
                    v_past_values.push_back(history_buffer_[value_name][i]);
                }

                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
                ImPlot::PlotLine(
                    std::string(value_name).append("_past").c_str(), t_past_values.data(), v_past_values.data(), v_past_values.size());
                ImPlot::PopStyleColor();
            }

            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0, 0.5f, 0.0f, 0.05f));
            for (auto traj : ego_policy.candidate_trajectories_)
            {
                std::vector<double> t_values;
                std::transform(traj.times.begin(), traj.times.end(), std::back_inserter(t_values), [&](auto t) { return time_s_ + t; });
                std::vector<double> values;
                std::transform(traj.states.colwise().begin(), traj.states.colwise().end(), std::back_inserter(values), getter_fn);
                ImPlot::PlotLine(value_name, t_values.data(), values.data(), values.size());
            }
            ImPlot::PopStyleColor();

            const auto &traj = ego_policy.get_last_solution();
            std::vector<double> t_values;
            std::transform(traj.times.begin(), traj.times.end(), std::back_inserter(t_values), [&](auto t) { return time_s_ + t; });
            std::vector<double> values;
            std::transform(traj.states.colwise().begin(), traj.states.colwise().end(), std::back_inserter(values), getter_fn);

            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
            ImPlot::PlotLine(std::string(value_name).append("_hola").c_str(), t_values.data(), values.data(), values.size());
            ImPlot::PopStyleColor();

            ImPlot::EndPlot();
        }
    };

    const auto make_action_plot = [&, this](auto title, auto value_name, auto getter_fn) {
        ImPlot::SetNextAxesToFit();
        if (ImPlot::BeginPlot(title))
        {
            ImPlot::SetupAxes("time[s]", value_name);
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0, 0.5f, 0.0f, 0.05f));
            for (auto traj : ego_policy.candidate_trajectories_)
            {
                std::vector<double> values;
                std::transform(traj.actions.colwise().begin(), traj.actions.colwise().end(), std::back_inserter(values), getter_fn);
                ImPlot::PlotLine(value_name, traj.times.data(), values.data(), values.size());
            }
            ImPlot::PopStyleColor();

            std::vector<double> values;
            std::transform(
                ego_policy.get_last_solution().actions.colwise().cbegin(),
                ego_policy.get_last_solution().actions.colwise().cend(),
                std::back_inserter(values),
                getter_fn);

            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
            ImPlot::PlotLine(
                std::string(value_name).append("_hola").c_str(), ego_policy.get_last_solution().times.data(), values.data(), values.size());
            ImPlot::PopStyleColor();
            ImPlot::EndPlot();
        }
    };

    EASY_BLOCK("Make plots");
    make_plot("Speed Plot", "speed_mps", [](const Ref<const typename Dynamics::State> &state) { return state[3]; });
    make_plot("Accel Plot", "accel_mpss", [](const Ref<const typename Dynamics::State> &state) { return state[4]; });
    make_plot("Yaw Plot", "yaw_rad", [](const Ref<const typename Dynamics::State> &state) { return state[2]; });
    make_plot("Steering Plot", "steering_rad", [](const Ref<const typename Dynamics::State> &state) { return state[5]; });
    make_action_plot("Jerk Plot", "jerk[mpsss]", [](const Ref<const typename Dynamics::Action> &action) {
        return 0.6 * std::tanh(action[0]);
    });
    make_action_plot("Steering Rate Plot", "srate[rad/s]", [](const Ref<const typename Dynamics::Action> &action) {
        return 0.1 * std::tanh(action[1]);
    });
    EASY_END_BLOCK;

    ImGui::End();
}

} // namespace mpex

MAGNUM_APPLICATION_MAIN(mpex::CEMMPCRacingApp)