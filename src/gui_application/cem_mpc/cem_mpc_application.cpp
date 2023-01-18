
#include <Magnum/Math/Color.h>
#include <Magnum/SceneGraph/Object.h>
#include <easy/profiler.h>

#include "base_application/base_application.hpp"
#include "cem_mpc/cem_mpc_application.hpp"
#include "cem_mpc/intelligent_driver_model.hpp"
#include "common/types.hpp"
#include "third_party/implot/implot.h"

namespace mpex
{

namespace
{
const auto red_color = Magnum::Math::Color3(1.0f, 0.2f, 0.0f);
const auto blue_color = Magnum::Math::Color3(0.0f, 0.6f, 1.0f);
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

  std::vector<float> x_vals;
  std::vector<float> y_vals;
  std::vector<float> z_vals;
  for (int i{0}; i < 100; ++i)
  {
    x_vals.push_back(i * 2.0);
    y_vals.push_back(0.0);
    z_vals.push_back(0.0);
  }

  centerline_->set_xy(x_vals, y_vals, z_vals);

  x_vals.clear();
  y_vals.clear();
  for (int i{0}; i < 100; ++i)
  {
    x_vals.push_back(i * 2.0);
    y_vals.push_back(4.0);
  }
  left_boundary_->set_xy(x_vals, y_vals, z_vals);

  x_vals.clear();
  y_vals.clear();
  for (int i{0}; i < 100; ++i)
  {
    x_vals.push_back(i * 2.0);
    y_vals.push_back(-4.0);
  }
  right_boundary_->set_xy(x_vals, y_vals, z_vals);

  IntelligentDriverModel::Config config;
  config.v0 = 10.0;
  config.N = horizon;
  idm_ = IntelligentDriverModel(config);

  // Initialize current state for the simulation, kinematic bicycle.
  current_state_ = Dynamics::State();
  current_state_ << 0.0, 0.0, 0.0, 10.0, 0.0, 0.0;
  idm_state_ = IntelligentDriverModel::State();
  idm_state_ << -20.0, 5.0;

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
    auto &object = trajectory_entities_.get_objects().at(i);

    (*object)
        .resetTransformation()
        .scale(trajectory_entities_.get_vehicle_extent())
        .translate(Magnum::Vector3(0.0f, 0.0f, SCALE(1.5f))) // move half wheelbase forward
        .rotateY(Magnum::Math::Rad(static_cast<float>(new_state[2])))
        .translate(Magnum::Vector3(SCALE(new_state[1]), SCALE(time_s), SCALE(new_state[0])));
  }

  const auto set_xy_helper = [this](const auto &_trajectory, auto &path_entity, auto color)
  {
    std::vector<float> x_vals;
    std::vector<float> y_vals;
    std::vector<float> t_vals;
    for (int i = 0; i < _trajectory.states.cols(); ++i)
    {
      Dynamics::State new_state = _trajectory.states.col(i);
      auto time_s = _trajectory.times.at(i);

      x_vals.push_back(new_state[0]);
      y_vals.push_back(new_state[1]);
      t_vals.push_back(time_s);
    }

    path_entity.set_xy(x_vals, y_vals, t_vals, color);
  };

  auto max_iter = std::max_element(mpc_.costs_index_pair_.begin(), mpc_.costs_index_pair_.end());
  auto min_iter = std::min_element(mpc_.costs_index_pair_.begin(), mpc_.costs_index_pair_.end());
  auto max_cost = max_iter != mpc_.costs_index_pair_.end() ? max_iter->first : 1e9;
  auto min_cost = min_iter != mpc_.costs_index_pair_.end() ? min_iter->first : -1e9;

  assert(path_entities_.size() == mpc_.candidate_trajectories_.size());
  for (int i = 0; i < path_entities_.size(); ++i)
  {
    auto cost = mpc_.costs_index_pair_[i].first;
    auto color_value = (cost - min_cost) / (max_cost - min_cost);
    
    std::cout << "max: " << max_cost << " min: "  << min_cost << "cost: " << cost << " color: " << color_value << std::endl;
    auto color = Magnum::Math::Color3(1.0f, 0.0f, static_cast<float>(color_value));
    set_xy_helper(mpc_.candidate_trajectories_.at(i), *path_entities_.at(i), color);
  }

  typename IntelligentDriverModel::States lead_states =
      IntelligentDriverModel::States::Ones(IntelligentDriverModel::state_size, trajectory.states.cols());

  lead_states.row(0).array() = 1000.0f; // initialize x positions with high value but not too high.
  lead_states.row(1).array() = idm_.config_.v0;

  for (int i = 0; i < trajectory.states.cols(); ++i)
  {
    // if (trajectory.states(1, i) < -2.0f && trajectory.states(1, i) > -8.0f)
    if (true)
    {
      lead_states(0, i) = trajectory.states(0, i); // x positions
      lead_states(1, i) = trajectory.states(3, i); // speeds
    }
  }

  const IntelligentDriverModel::States idm_states = idm_.rollout(idm_state_, lead_states);
  for (int i = 0; i < idm_states.cols(); ++i)
  {
    IntelligentDriverModel::State new_state = idm_states.col(i);
    auto time_s = trajectory.times.at(i);
    auto &object = trajectory_entities_idm_.get_objects().at(i);

    (*object)
        .resetTransformation()
        .scale(trajectory_entities_idm_.get_vehicle_extent())
        .translate(Magnum::Vector3(0.0f, 0.0f, SCALE(1.5f))) // move half wheelbase forward
        .translate(Magnum::Vector3(SCALE(-5.0f), SCALE(time_s), SCALE(new_state[0])));
  }

  auto idm_action = idm_.get_action(idm_state_, lead_states.col(0));

  current_state_ = Dynamics::step_(current_state_, current_action);
  idm_state_ = SingleIntegrator::step_(idm_state_, idm_action);

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
    current_state_ << 0.0, 0.0, 0.0, 10.0, 0.0, 0.0;
    idm_state_ << -20.0, 5.0;
    redraw();
  }

  if (ImGui::Button("Run MPC"))
  {
    is_running_ = is_running_ ? false : true;
    redraw();
  }

  ImGui::SliderInt("Num Iterations", &mpc_.get_num_iters_mutable(), 1, 100);

  double v_min = 0.0;
  double v_max = 10.0;
  ImGui::SliderScalarN("Cost Weights - States", ImGuiDataType_Double, &mpc_.cost_function_.w_s_, 6, &v_min, &v_max, "%.3f", 0);
  ImGui::SliderScalarN("Cost Weights - Actions", ImGuiDataType_Double, &mpc_.cost_function_.w_a_, 2, &v_min, &v_max, "%.3f", 0);
  ImGui::SliderScalarN("Terminal Cost Weights - States", ImGuiDataType_Double, &mpc_.cost_function_.W_s_, 6, &v_min, &v_max, "%.3f", 0);
  ImGui::SliderScalarN("TerminalCost Weights - Actions", ImGuiDataType_Double, &mpc_.cost_function_.W_a_, 2, &v_min, &v_max, "%.3f", 0);

  double v_min_r = -100.0;
  double v_max_r = 100.0;
  ImGui::SliderScalarN("Ref values - States", ImGuiDataType_Double, &mpc_.cost_function_.r_s_, 6, &v_min_r, &v_max_r, "%.3f", 0);
  ImGui::SliderScalarN("Ref values - Actions", ImGuiDataType_Double, &mpc_.cost_function_.r_a_, 2, &v_min_r, &v_max_r, "%.3f", 0);
  ImGui::SliderScalarN("Terminal Ref values - States", ImGuiDataType_Double, &mpc_.cost_function_.R_s_, 6, &v_min_r, &v_max_r, "%.3f", 0);
  ImGui::SliderScalarN("Terminal Ref values - Actions", ImGuiDataType_Double, &mpc_.cost_function_.R_a_, 2, &v_min_r, &v_max_r, "%.3f", 0);

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