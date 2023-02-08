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
#include "cem_mpc/cem_mpc_application.hpp"
#include "cem_mpc/intelligent_driver_model.hpp"
#include "common/dynamics.h"
#include "common/types.hpp"
#include "environment/lane_map.hpp"
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

void draw_candidate_paths(CEM_MPC<EigenKinematicBicycle> &mpc, std::vector<std::shared_ptr<Graphics::LineEntity>> &path_entities)
{
    const auto set_xy_helper = [](const auto &candidate_trajectory, auto &path_entity, auto color) {
        std::vector<double> x_vals;
        std::vector<double> y_vals;
        std::vector<double> z_vals;
        for (int i = 0; i < candidate_trajectory.states.cols(); ++i)
        {
            const EigenKinematicBicycle::State new_state = candidate_trajectory.states.col(i);
            x_vals.push_back(new_state[0]);
            y_vals.push_back(new_state[1]);
            z_vals.push_back(new_state[3]);
        }
        path_entity.set_xy(x_vals.size(), x_vals.data(), y_vals.data(), z_vals.data(), color);
    };

    const auto max_iter = std::max_element(mpc.costs_index_pair_.begin(), mpc.costs_index_pair_.end());
    const auto min_iter = std::min_element(mpc.costs_index_pair_.begin(), mpc.costs_index_pair_.end());
    const auto max_cost = max_iter != mpc.costs_index_pair_.end() ? max_iter->first : 1e9;
    const auto min_cost = min_iter != mpc.costs_index_pair_.end() ? min_iter->first : -1e9;

    assert(path_entities.size() == mpc.candidate_trajectories_.size());
    for (int i = 0; i < path_entities.size(); ++i)
    {
        float cost = mpc.costs_index_pair_[i].first;
        cost = (cost - min_cost) / (max_cost - min_cost); // normalize

        const auto exp_color_value_blue = std::exp(-10.0f * cost);
        const auto exp_color_value_red = 1.0f - exp_color_value_blue;
        const auto color = Magnum::Math::Color3(exp_color_value_red, 0.0f, exp_color_value_blue);

        auto &candidate_traj = mpc.candidate_trajectories_[i];
        set_xy_helper(candidate_traj, *path_entities.at(i), color);
    }
}

IntelligentDriverModelPolicy::States calc_idm_lead_states_from_trajectory(
    const IntelligentDriverModelPolicy &idm,
    const IntelligentDriverModelPolicy::State idm_state,
    const EigenKinematicBicycle::Trajectory &trajectory,
    const environment::Corridor lane)
{
    // In this case we need to explicitly define the type
    typename IntelligentDriverModelPolicy::States lead_states =
        IntelligentDriverModelPolicy::States::Ones(IntelligentDriverModelPolicy::state_size, trajectory.states.cols());

    lead_states.row(0).array() = 1000.0f; // initialize x positions with high value but not too high.
    lead_states.row(1).array() = idm.config_.v0;

    const auto agent_xy = Point2D(idm_state[0], idm_state[1]);
    const auto agent_dist_m = lane.centerline_.calc_curvilinear_coord(idm_state.head(2));

    for (size_t i{0}; i < trajectory.states.cols(); ++i)
    {
        const auto ego_xy = Point2D(trajectory.states(0, i), trajectory.states(1, i));
        const Vector2d current_ego_xy = trajectory.states.col(i).head(2);
        const auto ego_dist_m = lane.centerline_.calc_curvilinear_coord(current_ego_xy);

        if (lane.is_inside(ego_xy) && ego_dist_m > agent_dist_m)
        {
            lead_states(0, i) = trajectory.states(0, i); // x positions
            lead_states(1, i) = trajectory.states(3, i); // speeds
        }
    }
    return lead_states;
}

} // namespace

CEMMPCApplication::CEMMPCApplication(const Arguments &arguments) : Magnum::Examples::BaseApplication(arguments)
{

    ego_policy_ = CEM_MPC<EigenKinematicBicycle>(
        /* iters= */ 20,
        horizon,
        /* population= */ population,
        /* elites */ 10);

    ego_policy_.cost_function_ = std::make_shared<QuadraticCostFunction>();
    trajectory_entities_ = Graphics::TrajectoryEntities(scene_, horizon);
    trajectory_entities_idm_ = Graphics::TrajectoryEntities(scene_, horizon);

    _mesh = Magnum::MeshTools::compile(Magnum::Primitives::cubeWireframe());
    for (auto &object : trajectory_entities_.get_objects())
    {
        new Graphics::WireframeDrawable{*object, wireframe_shader_, _mesh, drawable_group_, red_color};
    }
    for (auto &object : trajectory_entities_idm_.get_objects())
    {
        new Graphics::WireframeDrawable{*object, wireframe_shader_, _mesh, drawable_group_, blue_color};
    }

    for (int i = 0; i < population; ++i)
    {
        path_entities_.push_back(std::make_shared<Graphics::LineEntity>(scene_));
        // new Graphics::
        //     VertexColorDrawable{*path_entities_.back()->object_ptr_, vertex_color_shader_, path_entities_.back()->mesh_,
        //     drawable_group_};
        new Graphics::FlatDrawable{*path_entities_.back()->object_ptr_, flat_shader_, path_entities_.back()->mesh_, drawable_group_};
    }

    lane_ = environment::create_line_helper(/* n_points */ 200);
    lane_entity_ = Graphics::LaneEntity(lane_, scene_, vertex_color_shader_, drawable_group_);

    IntelligentDriverModelPolicy::Config config;
    config.v0 = 10.0;
    config.N = horizon;
    actor_policy_ = IntelligentDriverModelPolicy(config);

    // Initialize current state for the simulation, kinematic bicycle.
    ego_state_ = Dynamics::State();
    ego_state_ << -10.0, 0.0, 0.0, 10.0, 0.0, 0.0;

    actor_state_ = SingleIntegrator::State();
    actor_state_ << -30.0, 5.0;

    history_buffer_["time_s"] = containers::Buffer<double>(n_buffer_capacity);
    history_buffer_["speed_mps"] = containers::Buffer<double>(n_buffer_capacity);
    history_buffer_["accel_mpss"] = containers::Buffer<double>(n_buffer_capacity);
    history_buffer_["yaw_rad"] = containers::Buffer<double>(n_buffer_capacity);
    history_buffer_["steering_rad"] = containers::Buffer<double>(n_buffer_capacity);

    runCEM();
}

void CEMMPCApplication::execute()
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

void CEMMPCApplication::runCEM()
{
    EASY_FUNCTION(profiler::colors::Red);

    auto trajectory = ego_policy_.solve(ego_state_, maybe_current_trajectory_);
    maybe_current_trajectory_ = trajectory;
    // auto trajectory = ego_policy_.solve(ego_state_);
    Dynamics::Action current_action = trajectory.actions.col(0);

    for (int i = 0; i < trajectory.states.cols(); ++i)
    {
        Dynamics::State new_state = trajectory.states.col(i);
        auto time_s = trajectory.times.at(i);
        // TODO: Create an SE2 utility function
        trajectory_entities_.set_state_at(i, new_state[0], new_state[1], new_state[2]);
    }

    draw_candidate_paths(ego_policy_, path_entities_);

    constexpr double idm_y_value_m = -5.0f;
    const auto lead_states = calc_idm_lead_states_from_trajectory(actor_policy_, actor_state_, trajectory, lane_);
    const auto idm_states = actor_policy_.rollout(actor_state_, lead_states);
    for (size_t i{0}; i < idm_states.cols(); ++i)
    {
        auto new_state = idm_states.col(i);
        trajectory_entities_idm_.set_state_at(i, new_state[0], idm_y_value_m, /* heading */ 0.0);
    }

    auto idm_action = actor_policy_.get_action(actor_state_, lead_states.col(0));

    Vector parameters = Vector::Zero(5);
    ego_state_ = Dynamics::step_(ego_state_, current_action, parameters);
    actor_state_ = SingleIntegrator::step_(actor_state_, idm_action);
    time_s_ += Dynamics::ts;

    history_buffer_["time_s"].push(time_s_);
    history_buffer_["speed_mps"].push(ego_state_[3]);
    history_buffer_["accel_mpss"].push(ego_state_[4]);
    history_buffer_["yaw_rad"].push(ego_state_[2]);
    history_buffer_["steering_rad"].push(ego_state_[5]);

    const auto ego_xy = Point2D(ego_state_[0], ego_state_[1]);
    EASY_END_BLOCK;
}

void CEMMPCApplication::show_menu()
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
        actor_state_ << -30.0, 5.0;
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

    auto& config_mutable = ego_policy_.get_config_mutable();
    ImGui::SliderInt("Num Iterations", &config_mutable.num_iters, 1, 100);
    ImGui::SliderInt("Num Elites", &config_mutable.elites, 1, 20);
    ImGui::SliderInt("Num Population", &config_mutable.population, 8, 2048);

    auto maybe_quadratic_cost_function_ptr = std::dynamic_pointer_cast<QuadraticCostFunction>(ego_policy_.cost_function_);
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

    const auto make_plot = [this](auto title, auto value_name, auto getter_fn) {
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
            for (auto traj : this->ego_policy_.candidate_trajectories_)
            {
                std::vector<double> t_values;
                std::transform(traj.times.begin(), traj.times.end(), std::back_inserter(t_values), [&](auto t) { return time_s_ + t; });
                std::vector<double> values;
                std::transform(traj.states.colwise().begin(), traj.states.colwise().end(), std::back_inserter(values), getter_fn);
                ImPlot::PlotLine(value_name, t_values.data(), values.data(), values.size());
            }
            ImPlot::PopStyleColor();

            const auto &traj = ego_policy_.get_last_solution();
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

    const auto make_action_plot = [this](auto title, auto value_name, auto getter_fn) {
        ImPlot::SetNextAxesToFit();
        if (ImPlot::BeginPlot(title))
        {
            ImPlot::SetupAxes("time[s]", value_name);
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0, 0.5f, 0.0f, 0.05f));
            for (auto traj : this->ego_policy_.candidate_trajectories_)
            {
                std::vector<double> values;
                std::transform(traj.actions.colwise().begin(), traj.actions.colwise().end(), std::back_inserter(values), getter_fn);
                ImPlot::PlotLine(value_name, traj.times.data(), values.data(), values.size());
            }
            ImPlot::PopStyleColor();

            std::vector<double> values;
            std::transform(
                this->ego_policy_.get_last_solution().actions.colwise().cbegin(),
                this->ego_policy_.get_last_solution().actions.colwise().cend(),
                std::back_inserter(values),
                getter_fn);

            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
            ImPlot::PlotLine(
                std::string(value_name).append("_hola").c_str(),
                ego_policy_.get_last_solution().times.data(),
                values.data(),
                values.size());
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

    EASY_FUNCTION(profiler::colors::Blue);
    ImGui::SetNextWindowPos({500.0f, 50.0f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("IntelligentDriverModelPolicy", nullptr);

    const auto make_plot_idm = [this](auto title, auto value_name, auto getter_fn) {
        ImPlot::SetNextAxesToFit();
        if (ImPlot::BeginPlot(title))
        {
            ImPlot::SetupAxes("time[s]", value_name);

            std::vector<double> values;
            auto states = actor_policy_.get_states();
            for (int i = 0; i < states.cols(); i++)
            {
                values.push_back(states(1, i));
            }
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
            ImPlot::PlotLine(
                std::string(value_name).append("_hola").c_str(),
                this->ego_policy_.get_last_solution().times.data(),
                values.data(),
                values.size());
            ImPlot::PopStyleColor();

            ImPlot::EndPlot();
        }
    };

    make_plot_idm("Speed Plot", "speed[mps]", [](const Ref<const SingleIntegrator::State> &state) { return state[1]; });
    ImGui::End();
}

} // namespace mpex

MAGNUM_APPLICATION_MAIN(mpex::CEMMPCApplication)