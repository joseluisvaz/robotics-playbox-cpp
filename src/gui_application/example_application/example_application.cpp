
#include "example_application/example_application.hpp"
#include "base_application/base_application.hpp"

namespace RoboticsSandbox
{

TemplateApplication::TemplateApplication(const Arguments &arguments) : Magnum::Examples::BaseApplication(arguments)
{
  std::cout << "constructing..." << std::endl;
}

void TemplateApplication::execute()
{
  std::cout << "executing..." << std::endl;
}

void TemplateApplication::show_menu()
{
  ImGui::ShowDemoWindow();
}

} // namespace RoboticsSandbox

MAGNUM_APPLICATION_MAIN(RoboticsSandbox::TemplateApplication)