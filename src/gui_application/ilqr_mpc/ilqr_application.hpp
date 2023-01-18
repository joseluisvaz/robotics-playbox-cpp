#include <Magnum/Platform/Sdl2Application.h>

#include "base_application/base_application.hpp"
#include "common/types.hpp"
#include "ilqr_mpc/ilqr_mpc.hpp"

namespace mpex
{

using namespace Eigen;

class IlqrMain : public Magnum::Examples::BaseApplication
{

public:
  using Dynamics = EigenKinematicBicycle;

  explicit IlqrMain(const Arguments &arguments);

private:
  virtual void show_menu() final;
  virtual void execute() final;

  void run_ilqr();

  Magnum::GL::Mesh mesh_{Magnum::NoCreate};
  Graphics::TrajectoryEntities trajectory_entities_;

  iLQR_MPC ilqr_mpc;
  bool is_running_{false};

  Dynamics::State current_state_;
};

} // namespace mpex
