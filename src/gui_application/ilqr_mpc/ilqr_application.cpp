#include <chrono>
#include <thread>

#include <Magnum/Math/Color.h>
#include <easy/profiler.h>

#include "gui_application/base_application/base_application.hpp"
#include "gui_application/common/types.hpp"
#include "ilqr_application.hpp"
#include "ilqr_mpc.hpp"

namespace mpex
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
  trajectory_entities_ = Graphics::TrajectoryEntities(scene_, horizon);
  mesh_ = Magnum::MeshTools::compile(Magnum::Primitives::cubeWireframe());

  auto color = red; // first element is the current state and must be red.
  for (auto &object : trajectory_entities_.get_objects())
  {
    new Graphics::WireframeDrawable{*object, wireframe_shader_, mesh_, drawable_group_, color};
    color = gray; // the rest of the wireframes will be gray.
  }

  policy_ = IterativeLinearQuadraticRegulator(/* horizon */ horizon, /* iters */ 30, /* debug */ true);

  // Initialize current state for the simulation, kinematic bicycle.
  current_state_ = Dynamics::State();
  current_state_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
}

void IlqrMain::execute()
{
  if (is_running_)
  {
    run_ilqr();
    is_running_ = false;
  }
}

void IlqrMain::run_ilqr()
{
  EASY_FUNCTION(profiler::colors::Red);
  const auto run_policy = [&](auto &policy, auto &entities)
  {
    Dynamics::Trajectory trajectory = policy.solve(current_state_, maybe_current_trajectory_);
    maybe_current_trajectory_ = trajectory;
    Dynamics::Action current_action = trajectory.actions.col(1);

    for (int i = 0; i < trajectory.states.cols() - 1; ++i)
    {
      Dynamics::State new_state = trajectory.states.col(i);
      entities.set_state_at(i, new_state[0], new_state[1], new_state[2]);
    }
    return current_action;
  };

  auto current_action = run_policy(policy_, trajectory_entities_);

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

} // namespace mpex

MAGNUM_APPLICATION_MAIN(mpex::IlqrMain)