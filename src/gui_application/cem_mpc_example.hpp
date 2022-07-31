
#include <Magnum/GL/Mesh.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <memory>
#include <vector>

#include "gui_application/base_example.hpp"
#include "gui_application/cem_mpc.hpp"
#include "gui_application/graphics/graphics_objects.hpp"
#include "gui_application/types.hpp"

namespace RoboticsSandbox
{

using namespace Eigen;

class CEMMPCExample : public Magnum::Examples::BaseExample
{
  using Dynamics = EigenKinematicBicycle;

public:
  explicit CEMMPCExample(const Arguments &arguments);

private:
  virtual void show_menu();
  virtual void execute();

  void runCEM();

  Magnum::GL::Mesh _mesh{Magnum::NoCreate};
  Graphics::TrajectoryObjects trajectory_objects_;
  std::vector<std::shared_ptr<Graphics::PathObjects>> path_objects_;
  CEM_MPC<Dynamics> mpc_;

  // flags
  bool is_running_{false};
};

} // namespace RoboticsSandbox
