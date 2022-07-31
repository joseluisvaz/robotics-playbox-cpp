
#include "cem_mpc.hpp"
#include "gui_application/graphics_objects.hpp"
#include "gui_application/base_example.hh"

namespace RoboticsSandbox
{

class CEMMPCExample : public Magnum::Example::BaseExample
{

public:
  explicit CEMMPCExample(const Arguments &arguments);

private:
//   void runCEM();

  // Setup Imgui Menu
  void show_menu() override;

  // Execute application contents
  void execute() override;

  //   Graphics::TrajectoryObjects trajectory_objects_;
  //   CEM_MPC<EigenKinematicBicycle> mpc_;

  //   // flags
  //   bool is_running_{false};

  //   // temps
  //   std::vector<std::shared_ptr<Graphics::PathObjects>> path_objects_;
};

} // namespace RoboticsSandbox
