#include <Magnum/Platform/Sdl2Application.h>

#include "gui_application/base_example.hpp"
#include "ilqr_mpc.hpp"
#include "types.hpp"

namespace RoboticsSandbox
{

using namespace Eigen;

class IlqrMain : public Magnum::Examples::BaseExample
{

public:
  using Dynamics = EigenKinematicBicycle;

  explicit IlqrMain(const Arguments &arguments);

private:
  virtual void show_menu() final;
  virtual void execute() final;

  void run_ilqr();

  Magnum::GL::Mesh mesh_{Magnum::NoCreate};
  Graphics::TrajectoryObjects trajectory_objects_;

  iLQR_MPC ilqr_mpc;
  bool is_running_{false};
};

} // namespace RoboticsSandbox
