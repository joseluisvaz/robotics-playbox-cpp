
#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Magnum/GL/Mesh.h>
#include <Magnum/Platform/Sdl2Application.h>

#include "base_application/base_application.hpp"
#include "cem_mpc/cem_mpc.h"
#include "cem_mpc/intelligent_driver_model.hpp"
#include "common/containers.h"
#include "common/dynamics.h"
#include "common/types.hpp"
#include "environment/lane_map.hpp"
#include "graphics/graphics_objects.hpp"

namespace mpex
{

using namespace Eigen;

class CEMMPCRacingApp : public Magnum::Examples::BaseApplication
{
  using Dynamics = EigenKinematicBicycle;

public:
  explicit CEMMPCRacingApp(const Arguments &arguments);

private:
  virtual void show_menu();
  virtual void execute();

  void runCEM();

  // Visualization entities for plotting using Magnum
  Magnum::GL::Mesh _mesh{Magnum::NoCreate};
  Graphics::TrajectoryEntities trajectory_entities_;
  std::vector<std::shared_ptr<Graphics::LineEntity>> path_entities_;
  std::shared_ptr<Graphics::LineEntity> left_boundary_;
  std::shared_ptr<Graphics::LineEntity> centerline_;
  std::shared_ptr<Graphics::LineEntity> right_boundary_;

  // Flags to control the GUI
  bool is_running_{false};

  // Ego policy and state information, used to control the agent that is using the MPC
  CEM_MPC<Dynamics> ego_policy_;
  Dynamics::State ego_state_;

  // Enviroment information such as the lane and the corresponding lanes
  environment::Lane lane_;
  Graphics::LaneEntity lane_entity_;

  // State of the simulation in this case the time_s
  double time_s_;
  std::unordered_map<std::string, containers::Buffer<double>> history_buffer_;
  std::optional<Dynamics::Trajectory> maybe_current_trajectory_;
};

} // namespace mpex
