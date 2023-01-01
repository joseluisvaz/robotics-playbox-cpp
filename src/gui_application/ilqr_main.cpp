#include <chrono>
#include <thread>

#include <Magnum/Math/Color.h>
#include <easy/profiler.h>

#include "gui_application/base_example.hpp"
#include "gui_application/ilqr_main.hpp"
#include "gui_application/ilqr_mpc.hpp"
#include "gui_application/types.hpp"

namespace RoboticsSandbox
{

IlqrMain::IlqrMain(const Arguments &arguments) : Magnum::Examples::BaseExample(arguments)
{
  constexpr int horizon = 20;
  trajectory_objects_ = Graphics::TrajectoryObjects(_scene, horizon);
  mesh_ = Magnum::MeshTools::compile(Magnum::Primitives::cubeWireframe());
  for (auto &object : trajectory_objects_.get_objects())
  {
    new Graphics::VertexColorDrawable{*object, _vertexColorShader, mesh_, _drawables};
  }

  ilqr_mpc = iLQR_MPC(/* horizon */ horizon, /* iters */ 70);

  // Initialize current state for the simulation, kinematic bicycle.
  current_state_ = Dynamics::State();
  current_state_ << 1.0, 10.0, 0.0, 0.0, 0.0, 0.0;
}

void IlqrMain::execute()
{
  if (is_running_)
  {
    run_ilqr();
  }
}

void IlqrMain::run_ilqr()
{
  EASY_FUNCTION(profiler::colors::Red);

  Dynamics::Trajectory trajectory = ilqr_mpc.solve(current_state_);
  Dynamics::Action current_action = trajectory.actions.col(1);
  
  for (int i = 0; i < trajectory.states.cols() - 1; ++i)
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
  
  current_state_ = Dynamics::step_(current_state_, current_action);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
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

  if (ImGui::Button("Run MPC"))
  {
    is_running_ = is_running_ ? false : true;
    redraw();
  }
  ImGui::End();
}

} // namespace RoboticsSandbox

MAGNUM_APPLICATION_MAIN(RoboticsSandbox::IlqrMain)