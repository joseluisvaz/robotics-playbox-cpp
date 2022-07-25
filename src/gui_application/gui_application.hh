#pragma once

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Image.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/FunctionsBatch.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/MeshTools/Transform.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Primitives/Axis.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Grid.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/Shaders/FlatGL.h>
#include <Magnum/Shaders/VertexColorGL.h>
#include <Magnum/Trade/MeshData.h>

#include "cem_mpc.hpp"
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <iostream>

#ifndef SCALE
#define SCALE(a) (a / 10.0)
#endif

namespace Magnum
{
namespace Examples
{

using namespace GuiApplication;

using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
using Scene3D = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;
using namespace Math::Literals;

class TrajectoryObjects
{
public:
  TrajectoryObjects() = default;
  TrajectoryObjects(Scene3D &scene, const int horizon_points);

  const std::vector<std::shared_ptr<Object3D>> &get_objects() const;
  const Vector3 &get_vehicle_extent() const;

private:
  std::vector<std::shared_ptr<Object3D>> objects_;
  std::vector<std::shared_ptr<Object3D>> line_objects_;
  Vector3 vehicle_extent_;
};

class PathObjects
{
  struct Vertex
  {
    Vector3 position;
    Color3 color;
  };

public:
  PathObjects()
  {
    data_ = std::vector<Vertex>(20, Vertex{Vector3{1.0f, 1.0f, 1.0f}, Color3(1.0f, 1.0f, 0.0f)});

    std::copy(data_.begin(), data_.begin() + 20, data_new_);

    buffer_ = GL::Buffer();
    buffer_.setData(data_new_, GL::BufferUsage::StaticDraw);
    printf("hola");
    mesh_.setPrimitive(GL::MeshPrimitive::LineStrip)
        .setCount(20)
        .addVertexBuffer(buffer_, 0, Shaders::VertexColorGL3D::Position{}, Shaders::VertexColorGL3D::Color3{});
    printf("adios");
  };

  void set_path(std::vector<float> x, std::vector<float> y)
  {
    std::vector<Vertex> points;
    for (int i = 0; i < x.size(); i++)
    {
      points.push_back(Vertex{Vector3{SCALE(y[i]), 0.0f, SCALE(x[i])}, Color3(1.0f, 1.0f, 0.0f)});
    }

    std::copy(points.begin(), points.begin() + x.size(), data_new_);
    buffer_.setData(data_new_, GL::BufferUsage::StaticDraw);
    mesh_.setPrimitive(GL::MeshPrimitive::LineStrip)
        .setCount(20)
        .addVertexBuffer(buffer_, 0, Shaders::VertexColorGL3D::Position{}, Shaders::VertexColorGL3D::Color3{});
  };

  std::vector<Vertex> data_;
  Vertex data_new_[20];
  GL::Buffer buffer_{NoCreate};
  GL::Mesh mesh_{NoCreate};
  Shaders::VertexColorGL3D shader_{NoCreate};
};

class SandboxExample : public Platform::Application
{

  using Dynamics = EigenKinematicBicycle;

public:
  explicit SandboxExample(const Arguments &arguments);

private:
  Float depthAt(const Vector2i &windowPosition);
  Vector3 unproject(const Vector2i &windowPosition, Float depth) const;

  // Custom methods
  void resetCameraPosition();
  void runCEM();

  // Mouse handling for the application
  void viewportEvent(ViewportEvent &event) override;
  void keyPressEvent(KeyEvent &event) override;
  void mousePressEvent(MouseEvent &event) override;
  void mouseReleaseEvent(MouseEvent &event) override;
  void mouseMoveEvent(MouseMoveEvent &event) override;
  void mouseScrollEvent(MouseScrollEvent &event) override;

  // Setup Imgui Menu
  void show_menu();

  // Draw event
  void drawEvent() override;

  ImGuiIntegration::Context _imgui{NoCreate};
  Shaders::VertexColorGL3D _vertexColorShader{NoCreate};
  Shaders::FlatGL3D _flatShader{NoCreate};
  GL::Mesh _mesh{NoCreate}, _grid{NoCreate}, _origin_axis{NoCreate};

  Scene3D _scene;
  SceneGraph::DrawableGroup3D _drawables;
  Object3D *_cameraObject;
  SceneGraph::Camera3D *_camera;

  Float _lastDepth;
  Vector2i _lastPosition{-1};
  Vector3 _rotationPoint, _translationPoint;

  // temp trajectory
  EigenKinematicBicycle::Trajectory _trajectory;
  TrajectoryObjects trajectory_objects_;
  CEM_MPC<EigenKinematicBicycle> mpc_;

  // flags
  bool is_running_{false};

  // temps
  std::shared_ptr<PathObjects> path_objects_;
};

class VertexColorDrawable : public SceneGraph::Drawable3D
{
public:
  explicit VertexColorDrawable(Object3D &object, Shaders::VertexColorGL3D &shader, GL::Mesh &mesh,
                               SceneGraph::DrawableGroup3D &drawables)
      : SceneGraph::Drawable3D{object, &drawables}, _shader(shader), _mesh(mesh)
  {
  }

  void draw(const Matrix4 &transformation, SceneGraph::Camera3D &camera)
  {
    _shader.setTransformationProjectionMatrix(camera.projectionMatrix() * transformation).draw(_mesh);
  }

private:
  Shaders::VertexColorGL3D &_shader;
  GL::Mesh &_mesh;
};

class FlatDrawable : public SceneGraph::Drawable3D
{
public:
  explicit FlatDrawable(Object3D &object, Shaders::FlatGL3D &shader, GL::Mesh &mesh,
                        SceneGraph::DrawableGroup3D &drawables)
      : SceneGraph::Drawable3D{object, &drawables}, _shader(shader), _mesh(mesh)
  {
  }

  void draw(const Matrix4 &transformation, SceneGraph::Camera3D &camera)
  {
    _shader
        .setColor(0x747474_rgbf) // gray color
        .setTransformationProjectionMatrix(camera.projectionMatrix() * transformation)
        .draw(_mesh);
  }

private:
  Shaders::FlatGL3D &_shader;
  GL::Mesh &_mesh;
};

} // namespace Examples
} // namespace Magnum
