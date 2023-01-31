
#pragma once

#include <Magnum/GL/Mesh.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <memory>
#include <vector>

#include "base_application/base_application.hpp"
#include "cem_mpc/cem_mpc.h"
#include "cem_mpc/intelligent_driver_model.hpp"
#include "common/types.hpp"
#include "environment/lane_map.hpp"
#include "graphics/graphics_objects.hpp"
#include "common/containers.h"

namespace mpex
{

using namespace Eigen;

class CEMMPCApplication : public Magnum::Examples::BaseApplication
{
  using Dynamics = EigenKinematicBicycle;

public:
  explicit CEMMPCApplication(const Arguments &arguments);

private:
  virtual void show_menu();
  virtual void execute();

  void runCEM();

  // Visualization
  Magnum::GL::Mesh _mesh{Magnum::NoCreate};
  Graphics::TrajectoryEntities trajectory_entities_;
  Graphics::TrajectoryEntities trajectory_entities_idm_;
  std::vector<std::shared_ptr<Graphics::LineEntity>> path_entities_;

  std::shared_ptr<Graphics::LineEntity> left_boundary_;
  std::shared_ptr<Graphics::LineEntity> centerline_;
  std::shared_ptr<Graphics::LineEntity> right_boundary_;

  // Algorithm
  CEM_MPC<Dynamics> mpc_;
  IntelligentDriverModel idm_;

  // flags
  bool is_running_{false};

  // state of the vehicle
  Dynamics::State current_state_;
  IntelligentDriverModel::State idm_state_;
  double time_s_;

  // enviroment
  environment::Lane lane_;
  Graphics::LaneEntity lane_entity_;

  // history buffer of states and actions
  std::unordered_map<std::string, containers::Buffer<double>> history_buffer_;
};

} // namespace mpex
