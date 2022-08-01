#include <Magnum/Platform/Sdl2Application.h>

#include "gui_application/base_example.hpp"

namespace RoboticsSandbox
{

using namespace Eigen;

class TemplateExample : public Magnum::Examples::BaseExample
{

public:
  explicit TemplateExample(const Arguments &arguments);

private:
  virtual void show_menu();
  virtual void execute();
};

} // namespace RoboticsSandbox
