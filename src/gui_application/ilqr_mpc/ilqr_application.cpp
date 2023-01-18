#include <chrono>
#include <thread>

#include <Magnum/Math/Color.h>
#include <easy/profiler.h>

#include "gui_application/base_application/base_application.hpp"
#include "gui_application/common/types.hpp"
#include "ilqr_application.hpp"
#include "ilqr_mpc.hpp"

namespace RoboticsSandbox
{

namespace
{

const auto red = Magnum::Math::Color3(1.0f, 0.2f, 0.0f);
const auto blue = Magnum::Math::Color3(0.0f, 0.6f, 1.0f);
const auto gray = Magnum::Math::Color3(0.4f, 0.4f, 0.4f);

} // namespace

IlqrMain::IlqrMain(const Arguments &arguments) : Magnum::Examples::BaseApplication(arguments)
{
  constexpr int horizon = 20;
  trajectory_objects_ = Graphics::TrajectoryObjects(_scene, horizon);
  mesh_ = Magnum::MeshTools::compile(Magnum::Primitives::cubeWireframe());

  auto color = red; // first element is the current state and must be red.
  for (auto &object : trajectory_objects_.get_objects())
  {
    new Graphics::WireframeDrawable{*object, _wireframe_shader, mesh_, _drawables, color};
    color = gray; // the rest of the wireframes will be gray.
  }

  ilqr_mpc = iLQR_MPC(/* horizon */ horizon, /* iters */ 20);

  // Initialize current state for the simulation, kinematic bicycle.
  current_state_ = Dynamics::State();
  current_state_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
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
  // std::this_thread::sleep_for(std::chrono::milliseconds(100));
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