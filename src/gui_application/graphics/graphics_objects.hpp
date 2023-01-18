#pragma once

#include <Corrade/Containers/ArrayViewStl.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/Shaders/FlatGL.h>
#include <Magnum/Shaders/MeshVisualizerGL.h>
#include <Magnum/Shaders/VertexColorGL.h>
#include <Magnum/Trade/MeshData.h>

#include <memory>
#include <vector>

#ifndef SCALE
#define SCALE(a) (a / 10.0f)
#endif

namespace mpex::Graphics
{

using namespace Magnum;
using namespace Math::Literals;
using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
using Scene3D = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

class TrajectoryEntities
{
public:
  TrajectoryEntities() = default;
  TrajectoryEntities(Scene3D &scene, const int horizon_points);

  const std::vector<std::shared_ptr<Object3D>> &get_objects() const;
  const Magnum::Vector3 &get_vehicle_extent() const;

private:
  std::vector<std::shared_ptr<Object3D>> objects_;
  std::vector<std::shared_ptr<Object3D>> line_objects_;
  Magnum::Vector3 vehicle_extent_;
};

class LineEntity
{
  struct Vertex
  {
    Magnum::Vector3 position;
    Color3 color;
  };

public:
  LineEntity() = delete;
  LineEntity(Scene3D &scene);

  void set_xy(std::vector<float> &x, std::vector<float> &y, std::vector<float> &z, const Color3 &color = Color3(1.0f, 1.0f, 0.0f));

  std::vector<Vertex> data_;
  GL::Mesh mesh_;
  GL::Buffer buffer_;
  std::shared_ptr<Object3D> object_ptr_;
};

class VertexColorDrawable : public SceneGraph::Drawable3D
{
public:
  explicit VertexColorDrawable(Object3D &object, Shaders::VertexColorGL3D &shader, GL::Mesh &mesh, SceneGraph::DrawableGroup3D &drawables);

  void draw(const Magnum::Matrix4 &transformation, SceneGraph::Camera3D &camera);

private:
  Shaders::VertexColorGL3D &_shader;
  GL::Mesh &_mesh;
};

class WireframeDrawable : public SceneGraph::Drawable3D
{
public:
  explicit WireframeDrawable(
      Object3D &object,
      Shaders::MeshVisualizerGL3D &shader,
      GL::Mesh &mesh,
      SceneGraph::DrawableGroup3D &drawables,
      const Color3 &color = Color3(1.0f, 1.0f, 0.0f));

  void draw(const Magnum::Matrix4 &transformation, SceneGraph::Camera3D &camera);

private:
  Shaders::MeshVisualizerGL3D &_shader;
  GL::Mesh &_mesh;
  Color3 _color;
};

class FlatDrawable : public SceneGraph::Drawable3D
{
public:
  explicit FlatDrawable(Object3D &object, Shaders::FlatGL3D &shader, GL::Mesh &mesh, SceneGraph::DrawableGroup3D &drawables);

  void draw(const Magnum::Matrix4 &transformation, SceneGraph::Camera3D &camera);

private:
  Shaders::FlatGL3D &_shader;
  GL::Mesh &_mesh;
};

} // namespace mpex::Graphics
