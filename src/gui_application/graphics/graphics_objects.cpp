
#include "gui_application/graphics/graphics_objects.hpp"

namespace RoboticsSandbox::Graphics
{

TrajectoryObjects::TrajectoryObjects(Scene3D &scene, const int horizon_points)
{
  constexpr float avg_length{SCALE(4.5f)};
  constexpr float avg_width{SCALE(1.75f)};
  constexpr float avg_height{SCALE(1.75f)};
  vehicle_extent_ = Vector3{avg_width, avg_height, avg_length};

  for (size_t i{0UL}; i < horizon_points; ++i)
  {
    auto cube = std::make_shared<Object3D>(&scene);
    objects_.emplace_back(std::move(cube));
  }
}

const std::vector<std::shared_ptr<Object3D>> &TrajectoryObjects::get_objects() const
{
  return objects_;
}

const Vector3 &TrajectoryObjects::get_vehicle_extent() const
{
  return vehicle_extent_;
}

PathObjects::PathObjects() : mesh_{GL::MeshPrimitive::LineStrip}
{
  mesh_.addVertexBuffer(buffer_, 0, Shaders::VertexColorGL3D::Position{}, Shaders::VertexColorGL3D::Color3{});
};

void PathObjects::set_path(std::vector<float> x, std::vector<float> y, std::vector<float> z, const Color3 &color)
{
  data_.clear();
  for (int i = 0; i < x.size(); i++)
  {
    data_.push_back(Vertex{Vector3{SCALE(y[i]), SCALE(z[i]), SCALE(x[i])}, color});
  }
  buffer_.setData(data_, GL::BufferUsage::StaticDraw);
  mesh_.setCount(data_.size());
};

VertexColorDrawable::VertexColorDrawable(Object3D &object, Shaders::VertexColorGL3D &shader, GL::Mesh &mesh,
                                         SceneGraph::DrawableGroup3D &drawables)
    : SceneGraph::Drawable3D{object, &drawables}, _shader(shader), _mesh(mesh)
{
}

void VertexColorDrawable::draw(const Magnum::Matrix4 &transformation, SceneGraph::Camera3D &camera)
{
  _shader.setTransformationProjectionMatrix(camera.projectionMatrix() * transformation).draw(_mesh);
}

FlatDrawable::FlatDrawable(Object3D &object, Shaders::FlatGL3D &shader, GL::Mesh &mesh,
                           SceneGraph::DrawableGroup3D &drawables)
    : SceneGraph::Drawable3D{object, &drawables}, _shader(shader), _mesh(mesh)
{
}

void FlatDrawable::draw(const Magnum::Matrix4 &transformation, SceneGraph::Camera3D &camera)
{
  _shader
      .setColor(0x747474_rgbf) // gray color
      .setTransformationProjectionMatrix(camera.projectionMatrix() * transformation)
      .draw(_mesh);
}

} // namespace RoboticsSandbox::Graphics
