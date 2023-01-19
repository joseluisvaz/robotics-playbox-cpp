
#include "gui_application/graphics/graphics_objects.hpp"

#include "Magnum/GL/DefaultFramebuffer.h"

namespace mpex::Graphics
{

TrajectoryEntities::TrajectoryEntities(Scene3D &scene, const int horizon_points)
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

const std::vector<std::shared_ptr<Object3D>> &TrajectoryEntities::get_objects() const
{
  return objects_;
}

const Vector3 &TrajectoryEntities::get_vehicle_extent() const
{
  return vehicle_extent_;
}

LineEntity::LineEntity(Scene3D &scene) : mesh_{GL::MeshPrimitive::LineStrip}
{
  mesh_.addVertexBuffer(buffer_, 0, Shaders::VertexColorGL3D::Position{}, Shaders::VertexColorGL3D::Color3{});
  object_ptr_ = std::make_shared<Object3D>(&scene);
};

void LineEntity::set_xy(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, const Color3 &color)
{
  data_.clear();
  for (int i = 0; i < x.size(); i++)
  {
    data_.push_back(
        Vertex{Vector3{SCALE(static_cast<float>(y[i])), SCALE(static_cast<float>(z[i])), SCALE(static_cast<float>(x[i]))}, color});
  }
  buffer_.setData(data_, GL::BufferUsage::StaticDraw);
  mesh_.setCount(data_.size());
};

VertexColorDrawable::VertexColorDrawable(
    Object3D &object, Shaders::VertexColorGL3D &shader, GL::Mesh &mesh, SceneGraph::DrawableGroup3D &drawables)
    : SceneGraph::Drawable3D{object, &drawables}, _shader(shader), _mesh(mesh)
{
}

void VertexColorDrawable::draw(const Magnum::Matrix4 &transformation, SceneGraph::Camera3D &camera)
{
  _shader.setTransformationProjectionMatrix(camera.projectionMatrix() * transformation).draw(_mesh);
}

WireframeDrawable::WireframeDrawable(
    Object3D &object, Shaders::MeshVisualizerGL3D &shader, GL::Mesh &mesh, SceneGraph::DrawableGroup3D &drawables, const Color3 &color)
    : SceneGraph::Drawable3D{object, &drawables}, _shader(shader), _mesh(mesh), _color(color)
{
}

void WireframeDrawable::draw(const Magnum::Matrix4 &transformation, SceneGraph::Camera3D &camera)
{
  _shader.setColor(_color)
      .setWireframeColor(_color)
      .setViewportSize(Vector2{GL::defaultFramebuffer.viewport().size()})
      .setTransformationProjectionMatrix(camera.projectionMatrix() * transformation)
      .draw(_mesh);
}

FlatDrawable::FlatDrawable(Object3D &object, Shaders::FlatGL3D &shader, GL::Mesh &mesh, SceneGraph::DrawableGroup3D &drawables)
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

} // namespace mpex::Graphics
