
#include <Magnum/GL/Mesh.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <memory>
#include <vector>

#include "base_application/base_application.hpp"
#include "cem_mpc/cem_mpc.hpp"
#include "cem_mpc/intelligent_driver_model.hpp"
#include "common/types.hpp"
#include "graphics/graphics_objects.hpp"

namespace RoboticsSandbox
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

  Magnum::GL::Mesh _mesh{Magnum::NoCreate};
  Graphics::TrajectoryObjects trajectory_objects_;
  Graphics::TrajectoryObjects trajectory_objects_idm_;
  std::vector<std::shared_ptr<Graphics::PathObjects>> path_objects_;
  CEM_MPC<Dynamics> mpc_;
  IntelligentDriverModel idm_;

  // flags
  bool is_running_{false};
};

} // namespace RoboticsSandbox
