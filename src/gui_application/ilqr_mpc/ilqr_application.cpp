#include <chrono>
#include <thread>

#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Color.h>
#include <easy/profiler.h>

#include "gui_application/base_application/base_application.hpp"
#include "gui_application/common/types.hpp"
#include "ilqr_application.hpp"
#include "ilqr_mpc.hpp"
#include "third_party/implot/implot.h"

namespace mpex
{

namespace
{

constexpr size_t n_buffer_capacity = 50;

using namespace Magnum::Math::Literals;
using std::size_t;

const auto red = Magnum::Math::Color3(1.0f, 0.2f, 0.0f);
const auto blue = Magnum::Math::Color3(0.0f, 0.6f, 1.0f);
const auto gray = Magnum::Math::Color3(0.4f, 0.4f, 0.4f);

} // namespace

IlqrMain::IlqrMain(const Arguments &arguments) : Magnum::Examples::BaseApplication(arguments)
{
  constexpr int horizon = 20;
  trajectory_entities_ = Graphics::TrajectoryEntities(scene_, horizon);
  mesh_ = Magnum::MeshTools::compile(Magnum::Primitives::cubeWireframe());

  const auto color = red;
  for (auto &object : trajectory_entities_.get_objects())
  {
    new Graphics::WireframeDrawable{*object, wireframe_shader_, mesh_, drawable_group_, color};
  }

  policy_ = IterativeLinearQuadraticRegulator(/* horizon */ horizon, /* iters */ 30, /* debug */ false);

  // Initialize current state for the simulation, kinematic bicycle.
  current_state_ = Dynamics::State();
  current_state_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  lane_ = environment::create_line_helper(/* n_points */ 200);
  lane_entity_ = Graphics::LaneEntity(lane_, scene_, vertex_color_shader_, drawable_group_);

  history_buffer_["time_s"] = containers::Buffer<double>(100);
  history_buffer_["speed_mps"] = containers::Buffer<double>(100);
  history_buffer_["accel_mpss"] = containers::Buffer<double>(100);
  history_buffer_["yaw_rad"] = containers::Buffer<double>(100);
  history_buffer_["steering_rad"] = containers::Buffer<double>(100);
}

void IlqrMain::execute()
{
  if (is_running_)
  {
    run_ilqr();
    camera_object_->resetTransformation()
        .translate(Magnum::Vector3::zAxis(SCALE(200.0f)))
        .translate(Magnum::Vector3::xAxis(SCALE(50.0f)))
        .rotateX(-90.0_degf)
        .rotateY(-90.0_degf)
        .translate(Magnum::Vector3::zAxis(SCALE(current_state_[0])))
        .translate(Magnum::Vector3::xAxis(SCALE(current_state_[1])));
    is_running_ = false;
  }
}

void IlqrMain::run_ilqr()
{
  EASY_FUNCTION(profiler::colors::Red);
  const auto run_policy = [&](auto &policy, auto &entities)
  {
    // Dynamics::Trajectory trajectory = policy.solve(current_state_, maybe_current_trajectory_);
    Dynamics::Trajectory trajectory = policy.solve(current_state_, {});
    maybe_current_trajectory_ = trajectory;
    Dynamics::Action current_action = trajectory.actions.col(0);

    for (int i = 0; i < trajectory.states.cols() - 1; ++i)
    {
      Dynamics::State new_state = trajectory.states.col(i);
      entities.set_state_at(i, new_state[0], new_state[1], new_state[2]);
    }
    return current_action;
  };

  auto current_action = run_policy(policy_, trajectory_entities_);

  Vector parameters = Vector::Zero(5);
  current_state_ = Dynamics::step_(current_state_, current_action, parameters);
  // std::this_thread::sleep_for(std::chrono::milliseconds(100));

  time_ += Dynamics::ts;
  history_buffer_["time_s"].push(time_);
  history_buffer_["speed_mps"].push(current_state_[3]);
  history_buffer_["accel_mpss"].push(current_state_[4]);
  history_buffer_["yaw_rad"].push(current_state_[2]);
  history_buffer_["steering_rad"].push(current_state_[5]);
}

void IlqrMain::show_menu()
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
    current_state_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    time_ = 0.0f;
    history_buffer_["time_s"] = containers::Buffer<double>(100);
    history_buffer_["speed_mps"] = containers::Buffer<double>(100);
    history_buffer_["accel_mpss"] = containers::Buffer<double>(100);
    history_buffer_["yaw_rad"] = containers::Buffer<double>(100);
    history_buffer_["steering_rad"] = containers::Buffer<double>(100);
    redraw();
  }

  if (ImGui::Button("Run MPC"))
  {
    is_running_ = is_running_ ? false : true;
    redraw();
  }

  if (!maybe_current_trajectory_)
  {
    ImGui::End();
    return;
  }

  const auto make_plot = [this](auto title, auto value_name, auto getter_fn)
  {
    ImPlot::SetNextAxesToFit();
    if (ImPlot::BeginPlot(title))
    {
      ImPlot::SetupAxes("time[s]", value_name);

      std::vector<double> values;
      std::transform(
          maybe_current_trajectory_->states.colwise().cbegin(),
          maybe_current_trajectory_->states.colwise().cend(),
          std::back_inserter(values),
          getter_fn);

      std::vector<double> t_values;
      const auto &times = maybe_current_trajectory_->times;
      for (int i{0}; i < times.size(); ++i)
      {
        t_values.push_back(time_ + times[i]);
      }

      for (auto &trajectory : policy_.debug_iterations_)
      {
        std::vector<double> _t_values;
        const auto &_times = trajectory.times;
        for (int i{0}; i < _times.size(); ++i)
        {
          _t_values.push_back(time_ + _times[i]);
        }
        std::vector<double> _values;
        std::transform(trajectory.states.colwise().cbegin(), trajectory.states.colwise().cend(), std::back_inserter(_values), getter_fn);

        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
        ImPlot::PlotLine(std::string(value_name).append("_hola").c_str(), _t_values.data(), _values.data(), _values.size());
        ImPlot::PopStyleColor();
      }

      // If there is information contained in the history buffer then plot it.
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
        ImPlot::PlotLine(std::string(value_name).append("_past").c_str(), t_past_values.data(), v_past_values.data(), v_past_values.size());
        ImPlot::PopStyleColor();
      }

      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
      ImPlot::PlotLine(std::string(value_name).append("_hola").c_str(), t_values.data(), values.data(), values.size());
      ImPlot::PopStyleColor();

      ImPlot::EndPlot();
    }
  };
  make_plot("Speed Plot", "speed_mps", [](const Ref<const typename Dynamics::State> &state) { return state[3]; });
  make_plot("Accel Plot", "accel_mpss", [](const Ref<const typename Dynamics::State> &state) { return state[4]; });
  make_plot("Yaw Plot", "yaw_rad", [](const Ref<const typename Dynamics::State> &state) { return state[2]; });
  make_plot("Steering Plot", "steering_rad", [](const Ref<const typename Dynamics::State> &state) { return state[5]; });

  ImGui::End();
}

} // namespace mpex

MAGNUM_APPLICATION_MAIN(mpex::IlqrMain)