#include "gui_application/base_example.hpp"

#include "gui_application/ilqr_main.hpp"
#include "gui_application/ilqr_mpc.hpp"
#include "gui_application/types.hpp"
#include "types.hpp"

namespace RoboticsSandbox
{

IlqrMain::IlqrMain(const Arguments &arguments) : Magnum::Examples::BaseExample(arguments)
{
  trajectory_objects_ = Graphics::TrajectoryObjects(_scene, 100);
  mesh_ = Magnum::MeshTools::compile(Magnum::Primitives::cubeWireframe());
  for (auto &object : trajectory_objects_.get_objects())
  {
    new Graphics::VertexColorDrawable{*object, _vertexColorShader, mesh_, _drawables};
  }

  auto ilqr_mpc = iLQR_MPC(/* horizon */ 20, /* iters */ 10);

  EigenKinematicBicycle::State state;
  state << 0.0, 0.0, 1.0, 1.0, 0.5, 0.5;
  ilqr_mpc.solve(state);
}

void IlqrMain::execute()
{
  // std::cout << "executing..." << std::endl;
}

void IlqrMain::show_menu()
{
  ImGui::ShowDemoWindow();
}

} // namespace RoboticsSandbox

MAGNUM_APPLICATION_MAIN(RoboticsSandbox::IlqrMain)