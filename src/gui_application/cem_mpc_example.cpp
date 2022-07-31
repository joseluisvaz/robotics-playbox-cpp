

#include "gui_application/cem_mpc_example.hpp"

namespace RoboticsSandbox
{

CEMMPCExample::CEMMPCExample(const Arguments &arguments) : Magnum::Examples::SandboxExample(arguments) {}

void show_menu()
{
  MAgnum::Examples::SandboxExample::show_menu();
}

void execute()
{
  MAgnum::Examples::SandboxExample::execute();
}

} // namespace RoboticsSandbox

MAGNUM_APPLICATION_MAIN(RoboticsSandbox::CEMMPCExample)