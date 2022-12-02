#include <Magnum/Platform/Sdl2Application.h>

#include "gui_application/base_example.hpp"

namespace RoboticsSandbox
{

using namespace Eigen;

class IlqrMain : public Magnum::Examples::BaseExample
{

public:
  explicit IlqrMain(const Arguments &arguments);

private:
  virtual void show_menu();
  virtual void execute();

  Magnum::GL::Mesh mesh_{Magnum::NoCreate};
  Graphics::TrajectoryObjects trajectory_objects_;
};

} // namespace RoboticsSandbox
