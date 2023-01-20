
#include <Magnum/Math/Color.h>
#include <Magnum/SceneGraph/Object.h>
#include <cstddef>
#include <easy/profiler.h>
#include <memory>

#include "base_application/base_application.hpp"
#include "cem_mpc/cem_mpc_application.hpp"
#include "cem_mpc/intelligent_driver_model.hpp"
#include "common/types.hpp"
#include "environment/lane_map.hpp"
#include "third_party/implot/implot.h"

namespace mpex
{

namespace
{

using geometry::P2D;
using geometry::Polyline2D;

const auto red_color = Magnum::Math::Color3(1.0f, 0.2f, 0.0f);
const auto blue_color = Magnum::Math::Color3(0.0f, 0.6f, 1.0f);

void draw_candidate_paths(CEM_MPC<EigenKinematicBicycle> &mpc, std::vector<std::shared_ptr<Graphics::LineEntity>> &path_entities)
{
  const auto set_xy_helper = [](const auto &candidate_trajectory, auto &path_entity, auto color)
  {
    std::vector<double> x_vals;
    std::vector<double> y_vals;
    std::vector<double> t_vals;
    for (int i = 0; i < candidate_trajectory.states.cols(); ++i)
    {
      const EigenKinematicBicycle::State new_state = candidate_trajectory.states.col(i);
      x_vals.push_back(new_state[0]);
      y_vals.push_back(new_state[1]);
      t_vals.push_back(candidate_trajectory.times.at(i));
    }
    path_entity.set_xy(x_vals, y_vals, t_vals, color);
  };

  const auto max_iter = std::max_element(mpc.costs_index_pair_.begin(), mpc.costs_index_pair_.end());
  const auto min_iter = std::min_element(mpc.costs_index_pair_.begin(), mpc.costs_index_pair_.end());
  const auto max_cost = max_iter != mpc.costs_index_pair_.end() ? max_iter->first : 1e9;
  const auto min_cost = min_iter != mpc.costs_index_pair_.end() ? min_iter->first : -1e9;
  assert(path_entities.size() == mpc.candidate_trajectories_.size());
  for (int i = 0; i < path_entities.size(); ++i)
  {
    const auto cost = mpc.costs_index_pair_[i].first;
    const auto exp_color_value_blue = std::exp(-10.0f * cost);
    const auto exp_color_value_red = 1.0f - exp_color_value_blue;
    const auto color = Magnum::Math::Color3(exp_color_value_red, 0.0f, exp_color_value_blue);
    set_xy_helper(mpc.candidate_trajectories_.at(i), *path_entities.at(i), color);
  }
}

IntelligentDriverModel::States calc_idm_lead_states_from_trajectory(
    const IntelligentDriverModel &idm,
    const IntelligentDriverModel::State &idm_state,
    const EigenKinematicBicycle::Trajectory &trajectory,
    const environment::Lane lane)
{
  // In this case we need to explicitly define the type
  typename IntelligentDriverModel::States lead_states =
      IntelligentDriverModel::States::Ones(IntelligentDriverModel::state_size, trajectory.states.cols());

  lead_states.row(0).array() = 1000.0f; // initialize x positions with high value but not too high.
  lead_states.row(1).array() = idm.config_.v0;

  const auto agent_xy = P2D(idm_state[0], idm_state[1]);
  const auto agent_dist_m = lane.centerline_.calc_progress_coord(agent_xy);

  for (size_t i{0}; i < trajectory.states.cols(); ++i)
  {
    const auto ego_xy = P2D(trajectory.states(0, i), trajectory.states(1, i));
    const auto ego_dist_m = lane.centerline_.calc_progress_coord(ego_xy);

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

  constexpr int horizon = 20;
  constexpr int population = 1024;
  mpc_ = CEM_MPC<EigenKinematicBicycle>(
      /* iters= */ 20,
      horizon,
      /* population= */ population,
      /* elites */ 16);

  mpc_.cost_function_ = std::make_shared<QuadraticCostFunction>();

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
    new Graphics::
        VertexColorDrawable{*path_entities_.back()->object_ptr_, vertex_color_shader_, path_entities_.back()->mesh_, drawable_group_};
  }

  centerline_ = std::make_shared<Graphics::LineEntity>(scene_);
  new Graphics::VertexColorDrawable{*centerline_->object_ptr_, vertex_color_shader_, centerline_->mesh_, drawable_group_};
  left_boundary_ = std::make_shared<Graphics::LineEntity>(scene_);
  new Graphics::VertexColorDrawable{*left_boundary_->object_ptr_, vertex_color_shader_, left_boundary_->mesh_, drawable_group_};
  right_boundary_ = std::make_shared<Graphics::LineEntity>(scene_);
  new Graphics::VertexColorDrawable{*right_boundary_->object_ptr_, vertex_color_shader_, right_boundary_->mesh_, drawable_group_};

  std::vector<double> x_vals;
  std::vector<double> y_vals;
  std::vector<double> z_vals;
  for (int i{0}; i < 100; ++i)
  {
    x_vals.push_back(i * 2.0);
    y_vals.push_back(0.0);
    z_vals.push_back(0.0);
  }

  auto centerline = Polyline2D(x_vals, y_vals);
  const auto gray_color = Magnum::Math::Color3(0.63f, 0.63f, 0.63f);
  centerline_->set_xy(x_vals, y_vals, z_vals, gray_color);

  x_vals.clear();
  y_vals.clear();
  for (int i{0}; i < 100; ++i)
  {
    x_vals.push_back(i * 2.0);
    y_vals.push_back(4.0);
  }
  auto left_boundary = Polyline2D(x_vals, y_vals);
  left_boundary_->set_xy(x_vals, y_vals, z_vals, gray_color);

  x_vals.clear();
  y_vals.clear();
  for (int i{0}; i < 100; ++i)
  {
    x_vals.push_back(i * 2.0);
    y_vals.push_back(-4.0);
  }
  auto right_boundary = Polyline2D(x_vals, y_vals);
  right_boundary_->set_xy(x_vals, y_vals, z_vals, gray_color);

  lane_ = environment::Lane(centerline, left_boundary, right_boundary);

  IntelligentDriverModel::Config config;
  config.v0 = 10.0;
  config.N = horizon;
  idm_ = IntelligentDriverModel(config);

  // Initialize current state for the simulation, kinematic bicycle.
  current_state_ = Dynamics::State();
  current_state_ << -10.0, 0.0, 0.0, 10.0, 0.0, 0.0;
  idm_state_ = IntelligentDriverModel::State();
  idm_state_ << -30.0, 5.0;

  // run one iteration of CEM to show in the window
  runCEM();
}

void CEMMPCApplication::execute()
{
  if (is_running_)
  {
    runCEM();
  }
}

void CEMMPCApplication::runCEM()
{
  EASY_FUNCTION(profiler::colors::Red);

  Dynamics::Trajectory &trajectory = mpc_.execute(current_state_);
  Dynamics::Action current_action = trajectory.actions.col(0);

  for (int i = 0; i < trajectory.states.cols(); ++i)
  {
    Dynamics::State new_state = trajectory.states.col(i);
    auto time_s = trajectory.times.at(i);
    // TODO: Create an SE2 utility function
    trajectory_entities_.set_state_at(i, new_state[0], new_state[1], new_state[2]);
  }

  draw_candidate_paths(mpc_, path_entities_);

  constexpr double idm_y_value_m = -5.0f;
  const auto lead_states = calc_idm_lead_states_from_trajectory(idm_, idm_state_, trajectory, lane_);
  const auto idm_states = idm_.rollout(idm_state_, lead_states);
  for (size_t i{0}; i < idm_states.cols(); ++i)
  {
    auto new_state = idm_states.col(i);
    trajectory_entities_idm_.set_state_at(i, new_state[0], idm_y_value_m, new_state[1], trajectory.times.at(i));
  }

  auto idm_action = idm_.get_action(idm_state_, lead_states.col(0));

  current_state_ = Dynamics::step_(current_state_, current_action);
  idm_state_ = SingleIntegrator::step_(idm_state_, idm_action);

  const auto ego_xy = P2D(current_state_[0], current_state_[1]);
  EASY_END_BLOCK;
}

void CEMMPCApplication::show_menu()
{

  // ImGui::ShowDemoWindow();
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
    current_state_ << -10.0, 0.0, 0.0, 10.0, 0.0, 0.0;
    idm_state_ << -30.0, 5.0;
    redraw();
  }

  if (ImGui::Button("Run MPC"))
  {
    is_running_ = is_running_ ? false : true;
    redraw();
  }

  ImGui::SliderInt("Num Iterations", &mpc_.get_num_iters_mutable(), 1, 100);

  auto maybe_quadratic_cost_function_ptr = std::dynamic_pointer_cast<QuadraticCostFunction>(mpc_.cost_function_);
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

  const auto make_plot = [this](auto title, auto value_name, auto getter_fn)
  {
    ImPlot::SetNextAxesToFit();
    if (ImPlot::BeginPlot(title))
    {
      ImPlot::SetupAxes("time[s]", value_name);

      for (auto traj : this->mpc_.candidate_trajectories_)
      {
        std::vector<double> values;
        std::transform(traj.states.colwise().begin(), traj.states.colwise().end(), std::back_inserter(values), getter_fn);
        ImPlot::PlotLine(value_name, traj.times.data(), values.data(), values.size());
      }

      std::vector<double> values;
      std::transform(
          mpc_.get_trajectory().states.colwise().cbegin(),
          mpc_.get_trajectory().states.colwise().cend(),
          std::back_inserter(values),
          getter_fn);

      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
      ImPlot::
          PlotLine(std::string(value_name).append("_hola").c_str(), this->mpc_.get_trajectory().times.data(), values.data(), values.size());
      ImPlot::PopStyleColor();

      ImPlot::EndPlot();
    }
  };

  const auto make_action_plot = [this](auto title, auto value_name, auto getter_fn)
  {
    ImPlot::SetNextAxesToFit();
    if (ImPlot::BeginPlot(title))
    {
      ImPlot::SetupAxes("time[s]", value_name);
      for (auto traj : this->mpc_.candidate_trajectories_)
      {
        std::vector<double> values;
        std::transform(traj.actions.colwise().begin(), traj.actions.colwise().end(), std::back_inserter(values), getter_fn);
        ImPlot::PlotLine(value_name, traj.times.data(), values.data(), values.size());
      }

      std::vector<double> values;
      std::transform(
          this->mpc_.get_trajectory().actions.colwise().cbegin(),
          this->mpc_.get_trajectory().actions.colwise().cend(),
          std::back_inserter(values),
          getter_fn);

      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
      ImPlot::PlotLine(std::string(value_name).append("_hola").c_str(), mpc_.get_trajectory().times.data(), values.data(), values.size());
      ImPlot::PopStyleColor();
      ImPlot::EndPlot();
    }
  };

  EASY_BLOCK("Make plots");
  make_plot("Speed Plot", "speed[mps]", [](const Ref<const typename Dynamics::State> &state) { return state[3]; });
  make_plot("Accel Plot", "accel[mpss]", [](const Ref<const typename Dynamics::State> &state) { return state[4]; });
  make_plot("Yaw Plot", "yaw[rad]", [](const Ref<const typename Dynamics::State> &state) { return state[2]; });
  make_plot("Steering Plot", "steering[rad]", [](const Ref<const typename Dynamics::State> &state) { return state[5]; });
  make_action_plot(
      "Jerk Plot", "jerk[mpsss]", [](const Ref<const typename Dynamics::Action> &action) { return 0.6 * std::tanh(action[0]); });
  make_action_plot(
      "Steering Rate Plot", "srate[rad/s]", [](const Ref<const typename Dynamics::Action> &action) { return 0.1 * std::tanh(action[1]); });
  EASY_END_BLOCK;

  ImGui::End();

  EASY_FUNCTION(profiler::colors::Blue);
  ImGui::SetNextWindowPos({500.0f, 50.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.5f);
  ImGui::Begin("IntelligentDriverModel", nullptr);

  const auto make_plot_idm = [this](auto title, auto value_name, auto getter_fn)
  {
    ImPlot::SetNextAxesToFit();
    if (ImPlot::BeginPlot(title))
    {
      ImPlot::SetupAxes("time[s]", value_name);

      std::vector<double> values;
      auto states = idm_.get_states();
      for (int i = 0; i < states.cols(); i++)
      {
        values.push_back(states(1, i));
      }
      // std::transform(idm_.get_states().colwise().cbegin(), idm_.get_states().colwise().cend(),
      //                std::back_inserter(values), getter_fn);

      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
      ImPlot::
          PlotLine(std::string(value_name).append("_hola").c_str(), this->mpc_.get_trajectory().times.data(), values.data(), values.size());
      ImPlot::PopStyleColor();

      ImPlot::EndPlot();
    }
  };

  make_plot_idm("Speed Plot", "speed[mps]", [](const Ref<const SingleIntegrator::State> &state) { return state[1]; });
  ImGui::End();
}

} // namespace mpex

MAGNUM_APPLICATION_MAIN(mpex::CEMMPCApplication)