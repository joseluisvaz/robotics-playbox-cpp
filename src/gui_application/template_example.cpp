#include "gui_application/base_example.hpp"

#include "gui_application/template_example.hpp"

namespace RoboticsSandbox
{

TemplateExample::TemplateExample(const Arguments &arguments) : Magnum::Examples::BaseExample(arguments)
{
  std::cout << "constructing..." << std::endl;
}

void TemplateExample::execute()
{
  std::cout << "executing..." << std::endl;
}

void TemplateExample::show_menu()
{
  ImGui::ShowDemoWindow();
}

} // namespace RoboticsSandbox

MAGNUM_APPLICATION_MAIN(RoboticsSandbox::TemplateExample)