#include <Magnum/Math/Color.h>
#include <Magnum/Platform/Sdl2Application.h>

#include "base_application//base_application.hpp"

namespace mpex {

using namespace Eigen;

class TemplateApplication : public Magnum::Examples::BaseApplication
{
    struct Vertex
    {
        Magnum::Vector3 point;
        Magnum::Color3 color;
    };

  public:
    explicit TemplateApplication(const Arguments &arguments);

  private:
    virtual void show_menu();
    virtual void execute();

    std::vector<Vertex> data_;
    Magnum::GL::Mesh mesh_{Magnum::NoCreate};
};

} // namespace mpex
