
#include <Magnum/Math/Color.h>
#include <Magnum/SceneGraph/Object.h>
#include <easy/profiler.h>

#include "gui_application/base_example.hpp"
#include "gui_application/cem_mpc_example.hpp"
#include "gui_application/implot.h"
#include "intelligent_driver_model.hpp"
#include "types.hpp"

namespace RoboticsSandbox
{

namespace
{
const auto red_color = Magnum::Math::Color3(1.0f, 0.2f, 0.0f);
const auto blue_color = Magnum::Math::Color3(0.0f, 0.6f, 1.0f);
} // namespace

CEMMPCExample::CEMMPCExample(const Arguments &arguments) : Magnum::Examples::BaseExample(arguments)
{

  constexpr int horizon = 20;
  constexpr int population = 1024;
  mpc_ = CEM_MPC<EigenKinematicBicycle>(/* iters= */ 20, horizon,
                                        /* population= */ population,
                                        /* elites */ 16);

  trajectory_objects_ = Graphics::TrajectoryObjects(_scene, horizon);
  trajectory_objects_idm_ = Graphics::TrajectoryObjects(_scene, horizon);

  _mesh = Magnum::MeshTools::compile(Magnum::Primitives::cubeWireframe());
  for (auto &object : trajectory_objects_.get_objects())
  {
    new Graphics::WireframeDrawable{*object, _wireframe_shader, _mesh, _drawables, red_color};
  }
  for (auto &object : trajectory_objects_idm_.get_objects())
  {
    new Graphics::WireframeDrawable{*object, _wireframe_shader, _mesh, _drawables, blue_color};
  }

  for (int i = 0; i < population; ++i)
  {
    auto path_object = std::make_shared<Graphics::PathObjects>();
    path_objects_.push_back(path_object);
    auto vtx = new Magnum::Examples::BaseExample::Object3D{&_scene};
    new Graphics::VertexColorDrawable{*vtx, _vertexColorShader, path_object->mesh_, _drawables};
  }

  IntelligentDriverModel::Config config;
  config.v0 = 10.0;
  config.N = horizon;
  idm_ = IntelligentDriverModel(config);

  // run one iteration of CEM to show in the window
  runCEM();
}

void CEMMPCExample::execute()
{
  if (is_running_)
  {
    runCEM();
  }
}

void CEMMPCExample::runCEM()
{
  EASY_FUNCTION(profiler::colors::Red);

  Dynamics::State state = Dynamics::State::Zero();
  state[3] = 10.0;
  Dynamics::Trajectory &trajectory = mpc_.execute(state);

  for (int i = 0; i < trajectory.states.cols(); ++i)
  {
    Dynamics::State new_state = trajectory.states.col(i);
    auto time_s = trajectory.times.at(i);
    auto &object = trajectory_objects_.get_objects().at(i);

    (*object)
        .resetTransformation()
        .scale(trajectory_objects_.get_vehicle_extent())
        .translate(Magnum::Vector3(0.0f, 0.0f, SCALE(1.5f))) // move half wheelbase forward
        .rotateY(Magnum::Math::Rad(static_cast<float>(new_state[2])))
        .translate(Magnum::Vector3(SCALE(new_state[1]), SCALE(time_s), SCALE(new_state[0])));
  }

  const auto set_path_helper = [this](const auto &_trajectory, auto &_path_object) {
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

    _path_object.set_path(x_vals, y_vals, t_vals);
  };

  assert(path_objects_.size() == mpc_.candidate_trajectories_.size());
  for (int i = 0; i < path_objects_.size(); ++i)
  {
    set_path_helper(mpc_.candidate_trajectories_.at(i), *path_objects_.at(i));
  }

  typename IntelligentDriverModel::State this_state = IntelligentDriverModel::State::Zero();
  this_state[0] = -20.0f;
  this_state[1] = 10.0f;

  typename IntelligentDriverModel::States lead_states =
      IntelligentDriverModel::States::Ones(IntelligentDriverModel::state_size, trajectory.states.cols());
  lead_states.row(0).array() = 1000.0f;
  lead_states.row(1).array() = idm_.config_.v0;

  for (int i = 0; i < trajectory.states.cols(); ++i)
  {
    if (trajectory.states(1, i) < -2.0f && trajectory.states(1, i) > -8.0f)
    {
      lead_states(0, i) = trajectory.states(0, i);
      lead_states(1, i) = trajectory.states(3, i);
    }
  }

  const IntelligentDriverModel::States idm_states = idm_.rollout(this_state, lead_states);

  for (int i = 0; i < idm_states.cols(); ++i)
  {
    IntelligentDriverModel::State new_state = idm_states.col(i);
    auto time_s = trajectory.times.at(i);
    auto &object = trajectory_objects_idm_.get_objects().at(i);

    (*object)
        .resetTransformation()
        .scale(trajectory_objects_idm_.get_vehicle_extent())
        .translate(Magnum::Vector3(0.0f, 0.0f, SCALE(1.5f))) // move half wheelbase forward
        .translate(Magnum::Vector3(SCALE(-5.0f), SCALE(time_s), SCALE(new_state[0])));
  }

  EASY_END_BLOCK;
}

void CEMMPCExample::show_menu()
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

  if (ImGui::Button("Run MPC"))
  {
    is_running_ = is_running_ ? false : true;
    redraw();
  }

  ImGui::SliderInt("Num Iterations", &mpc_.get_num_iters_mutable(), 1, 100);

  double v_min = 0.0;
  double v_max = 10.0;
  ImGui::SliderScalarN("Cost Weights - States", ImGuiDataType_Double, &mpc_.cost_function_.w_s_, 6, &v_min, &v_max,
                       "%.3f", 0);
  ImGui::SliderScalarN("Cost Weights - Actions", ImGuiDataType_Double, &mpc_.cost_function_.w_a_, 2, &v_min, &v_max,
                       "%.3f", 0);
  ImGui::SliderScalarN("Terminal Cost Weights - States", ImGuiDataType_Double, &mpc_.cost_function_.W_s_, 6, &v_min,
                       &v_max, "%.3f", 0);
  ImGui::SliderScalarN("TerminalCost Weights - Actions", ImGuiDataType_Double, &mpc_.cost_function_.W_a_, 2, &v_min,
                       &v_max, "%.3f", 0);

  double v_min_r = -100.0;
  double v_max_r = 100.0;
  ImGui::SliderScalarN("Ref values - States", ImGuiDataType_Double, &mpc_.cost_function_.r_s_, 6, &v_min_r, &v_max_r,
                       "%.3f", 0);
  ImGui::SliderScalarN("Ref values - Actions", ImGuiDataType_Double, &mpc_.cost_function_.r_a_, 2, &v_min_r, &v_max_r,
                       "%.3f", 0);
  ImGui::SliderScalarN("Terminal Ref values - States", ImGuiDataType_Double, &mpc_.cost_function_.R_s_, 6, &v_min_r,
                       &v_max_r, "%.3f", 0);
  ImGui::SliderScalarN("Terminal Ref values - Actions", ImGuiDataType_Double, &mpc_.cost_function_.R_a_, 2, &v_min_r,
                       &v_max_r, "%.3f", 0);

  const auto make_plot = [this](auto title, auto value_name, auto getter_fn) {
    ImPlot::SetNextAxesToFit();
    if (ImPlot::BeginPlot(title))
    {
      ImPlot::SetupAxes("time[s]", value_name);

      for (auto traj : this->mpc_.candidate_trajectories_)
      {
        std::vector<double> values;
        std::transform(traj.states.colwise().begin(), traj.states.colwise().end(), std::back_inserter(values),
                       getter_fn);
        ImPlot::PlotLine(value_name, traj.times.data(), values.data(), values.size());
      }

      std::vector<double> values;
      std::transform(mpc_.get_trajectory().states.colwise().cbegin(), mpc_.get_trajectory().states.colwise().cend(),
                     std::back_inserter(values), getter_fn);

      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
      ImPlot::PlotLine(std::string(value_name).append("_hola").c_str(), this->mpc_.get_trajectory().times.data(),
                       values.data(), values.size());
      ImPlot::PopStyleColor();

      ImPlot::EndPlot();
    }
  };

  const auto make_action_plot = [this](auto title, auto value_name, auto getter_fn) {
    ImPlot::SetNextAxesToFit();
    if (ImPlot::BeginPlot(title))
    {
      ImPlot::SetupAxes("time[s]", value_name);
      for (auto traj : this->mpc_.candidate_trajectories_)
      {
        std::vector<double> values;
        std::transform(traj.actions.colwise().begin(), traj.actions.colwise().end(), std::back_inserter(values),
                       getter_fn);
        ImPlot::PlotLine(value_name, traj.times.data(), values.data(), values.size());
      }

      std::vector<double> values;
      std::transform(this->mpc_.get_trajectory().actions.colwise().cbegin(),
                     this->mpc_.get_trajectory().actions.colwise().cend(), std::back_inserter(values), getter_fn);

      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
      ImPlot::PlotLine(std::string(value_name).append("_hola").c_str(), mpc_.get_trajectory().times.data(),
                       values.data(), values.size());
      ImPlot::PopStyleColor();
      ImPlot::EndPlot();
    }
  };

  EASY_BLOCK("Make plots");
  make_plot("Speed Plot", "speed[mps]", [](const Ref<const typename Dynamics::State> &state) { return state[3]; });
  make_plot("Accel Plot", "accel[mpss]", [](const Ref<const typename Dynamics::State> &state) { return state[4]; });
  make_plot("Yaw Plot", "yaw[rad]", [](const Ref<const typename Dynamics::State> &state) { return state[2]; });
  make_plot("Steering Plot", "steering[rad]",
            [](const Ref<const typename Dynamics::State> &state) { return state[5]; });
  make_action_plot("Jerk Plot", "jerk[mpsss]",
                   [](const Ref<const typename Dynamics::Action> &action) { return 0.6 * std::tanh(action[0]); });
  make_action_plot("Steering Rate Plot", "srate[rad/s]",
                   [](const Ref<const typename Dynamics::Action> &action) { return 0.1 * std::tanh(action[1]); });
  EASY_END_BLOCK;

  ImGui::End();

  EASY_FUNCTION(profiler::colors::Blue);
  ImGui::SetNextWindowPos({500.0f, 50.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.5f);
  ImGui::Begin("IntelligentDriverModel", nullptr);

  const auto make_plot_idm = [this](auto title, auto value_name, auto getter_fn) {
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
      ImPlot::PlotLine(std::string(value_name).append("_hola").c_str(), this->mpc_.get_trajectory().times.data(),
                       values.data(), values.size());
      ImPlot::PopStyleColor();

      ImPlot::EndPlot();
    }
  };

  make_plot_idm("Speed Plot", "speed[mps]", [](const Ref<const SingleIntegrator::State> &state) { return state[1]; });
  ImGui::End();
}

} // namespace RoboticsSandbox

MAGNUM_APPLICATION_MAIN(RoboticsSandbox::CEMMPCExample)