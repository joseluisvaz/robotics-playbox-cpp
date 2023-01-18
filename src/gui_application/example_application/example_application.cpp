
#include "example_application/example_application.hpp"
#include "base_application/base_application.hpp"

namespace mpex
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

} // namespace mpex

MAGNUM_APPLICATION_MAIN(mpex::TemplateApplication)