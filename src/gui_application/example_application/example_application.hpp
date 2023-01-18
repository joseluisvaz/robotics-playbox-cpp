#include <Magnum/Platform/Sdl2Application.h>

#include "base_application/base_application.hpp"

namespace mpex
{

class TemplateApplication : public Magnum::Examples::BaseApplication
{

public:
  explicit TemplateApplication(const Arguments &arguments);

private:
  virtual void show_menu();
  virtual void execute();
};

} // namespace mpex
