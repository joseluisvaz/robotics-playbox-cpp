#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Math/Color.h>

#include "gui_application/base_example.hpp"

namespace RoboticsSandbox
{

using namespace Eigen;

class TsdfExample : public Magnum::Examples::BaseExample
{
  struct Vertex
  {
    Magnum::Vector3 point;
    Magnum::Color3 color;
  };

public:
  explicit TsdfExample(const Arguments &arguments);

private:
  virtual void show_menu();
  virtual void execute();

  std::vector<Vertex> data_;
  Magnum::GL::Mesh mesh_{Magnum::NoCreate};
};

} // namespace RoboticsSandbox
