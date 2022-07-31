
#include "gui_application/base_example.hpp"
#include "gui_application/implot.h"
#include <Magnum/SceneGraph/Object.h>
#include <easy/profiler.h>

#include "gui_application/cem_mpc_example.hpp"

namespace RoboticsSandbox
{

CEMMPCExample::CEMMPCExample(const Arguments &arguments) : Magnum::Examples::BaseExample(arguments)
{

  constexpr int horizon = 20;
  constexpr int population = 1024;
  mpc_ = CEM_MPC<EigenKinematicBicycle>(/* iters= */ 20, horizon,
                                        /* population= */ population,
                                        /* elites */ 16);

  trajectory_objects_ = Graphics::TrajectoryObjects(_scene, horizon);

  _mesh = Magnum::MeshTools::compile(Magnum::Primitives::cubeWireframe());
  for (auto &object : trajectory_objects_.get_objects())
  {
    new Graphics::VertexColorDrawable{*object, _vertexColorShader, _mesh, _drawables};
  }

  for (int i = 0; i < population; ++i)
  {
    auto path_object = std::make_shared<Graphics::PathObjects>();
    path_objects_.push_back(path_object);
    auto vtx = new Magnum::Examples::BaseExample::Object3D{&_scene};
    new Graphics::VertexColorDrawable{*vtx, _vertexColorShader, path_object->mesh_, _drawables};
  }

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

  EASY_BLOCK("Plotting All");
  for (int i = 0; i < trajectory.states.cols(); ++i)
  {
    Dynamics::State new_state = trajectory.states.col(i);
    auto time_s = trajectory.times.at(i);
    auto &object = trajectory_objects_.get_objects().at(i);

    EASY_BLOCK("Plotting");
    (*object)
        .resetTransformation()
        .scale(trajectory_objects_.get_vehicle_extent())
        .translate(Magnum::Vector3(0.0f, 0.0f, SCALE(1.5f))) // move half wheelbase forward
        .rotateY(Magnum::Math::Rad(new_state[2]))
        .translate(Magnum::Vector3(SCALE(new_state[1]), SCALE(time_s), SCALE(new_state[0])));
    EASY_END_BLOCK;
  }

  const auto set_path_helper = [this](const auto &_trajectory, auto &_path_object)
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

    _path_object.set_path(x_vals, y_vals, t_vals);
  };

  assert(path_objects_.size() == mpc_.candidate_trajectories_.size());
  for (int i = 0; i < path_objects_.size(); ++i)
  {
    set_path_helper(mpc_.candidate_trajectories_.at(i), *path_objects_.at(i));
  }

  EASY_END_BLOCK;
}

void CEMMPCExample::show_menu()
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

  if (ImGui::Button("Run MPC"))
  {
    is_running_ = is_running_ ? false : true;
    redraw();
  }

  ImGui::SliderInt("Num Iterations", &mpc_.get_num_iters_mutable(), 1, 100);
  ImGui::SliderFloat3("costs[x, y, yaw]", mpc_.cost_function_.state_slider_values_, 0.0f, 10.0f);
  ImGui::SliderFloat3("costs[speed, acc, steering]", mpc_.cost_function_.state_slider_values_2_, 0.0f, 10.0f);
  ImGui::SliderFloat2("costs[jerk, steering]", mpc_.cost_function_.action_slider_values_, 0.0f, 10.0f);
  ImGui::SliderFloat3("ref[x, y, speed]", mpc_.cost_function_.ref_values_, 0.0f, 10.0f);
  ImGui::SliderAngle("ref[]", &mpc_.cost_function_.ref_yaw_, 0.0f, 180.0f);

  ImGui::SliderFloat3("terminal costs[x, y, yaw]", mpc_.cost_function_.terminal_state_slider_values_, 0.0f, 10.0f);
  ImGui::SliderFloat3("terminal costs[speed, acc, steering]", mpc_.cost_function_.terminal_state_slider_values_2_, 0.0f,
                      10.0f);
  ImGui::SliderFloat2("terminal costs[jerk, steering]", mpc_.cost_function_.terminal_action_slider_values_, 0.0f,
                      10.0f);
  ImGui::SliderFloat3("terminal ref[x, y, speed]", mpc_.cost_function_.terminal_ref_values_, -100.0f, 100.0f);
  ImGui::SliderAngle("terminal ref[]", &mpc_.cost_function_.terminal_ref_yaw_, -180.0f, 180.0f);

  const auto make_plot = [this](auto title, auto value_name, auto getter_fn)
  {
    ImPlot::SetNextAxesToFit();
    if (ImPlot::BeginPlot(title))
    {
      ImPlot::SetupAxes("time[s]", value_name);

      for (auto traj : this->mpc_.candidate_trajectories_)
      {
        std::vector<float> values;
        std::transform(traj.states.colwise().begin(), traj.states.colwise().end(), std::back_inserter(values),
                       getter_fn);
        ImPlot::PlotLine(value_name, traj.times.data(), values.data(), values.size());
      }

      std::vector<float> values;
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

  const auto make_action_plot = [this](auto title, auto value_name, auto getter_fn)
  {
    ImPlot::SetNextAxesToFit();
    if (ImPlot::BeginPlot(title))
    {
      ImPlot::SetupAxes("time[s]", value_name);
      for (auto traj : this->mpc_.candidate_trajectories_)
      {
        std::vector<float> values;
        std::transform(traj.actions.colwise().begin(), traj.actions.colwise().end(), std::back_inserter(values),
                       getter_fn);
        ImPlot::PlotLine(value_name, traj.times.data(), values.data(), values.size());
      }

      std::vector<float> values;
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
  make_plot("Steering Plot", "steering[rad]", [](const Ref<const typename Dynamics::State> &state) { return state[5]; });
  make_action_plot("Jerk Plot", "jerk[mpsss]",
                   [](const Ref<const typename Dynamics::Action> &action) { return 0.6 * std::tanh(action[0]); });
  make_action_plot("Steering Rate Plot", "srate[rad/s]",
                   [](const Ref<const typename Dynamics::Action> &action) { return 0.1 * std::tanh(action[1]); });
  EASY_END_BLOCK;

  ImGui::End();
}

} // namespace RoboticsSandbox

MAGNUM_APPLICATION_MAIN(RoboticsSandbox::CEMMPCExample)