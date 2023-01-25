#pragma once

#include <Corrade/Containers/ArrayViewStl.h>
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

#include "gui_application/graphics/graphics_objects.hpp"
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <iostream>

#ifndef SCALE
#define SCALE(a) (a / 10.0)
#endif

namespace Magnum
{
namespace Examples
{

using namespace Math::Literals;
using namespace mpex;

class BaseApplication : public Platform::Application
{

public:
  using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
  using Scene3D = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

  explicit BaseApplication(const Arguments &arguments);

  void resetCameraPosition();

  // functions to overwrite to create new applications/examples.
  virtual void show_menu() = 0;
  virtual void execute() = 0;

  Scene3D scene_;
  SceneGraph::DrawableGroup3D drawable_group_;

  // Shader to render meshes with position and color information
  Shaders::VertexColorGL3D vertex_color_shader_{NoCreate};

  // Shader to render meshes with color information
  Shaders::MeshVisualizerGL3D wireframe_shader_{NoCreate};

  // Shader to render flat meshes with fixed color information
  Shaders::FlatGL3D flat_shader_{NoCreate};

  // Camera object
  Object3D *camera_object_;

private:
  Float depthAt(const Vector2i &windowPosition);
  Vector3 unproject(const Vector2i &windowPosition, Float depth) const;

  // Custom methods
  void runCEM();

  // Mouse handling for the application
  void viewportEvent(ViewportEvent &event) override;
  void keyPressEvent(KeyEvent &event) override;
  void mousePressEvent(MouseEvent &event) override;
  void mouseReleaseEvent(MouseEvent &event) override;
  void mouseMoveEvent(MouseMoveEvent &event) override;
  void mouseScrollEvent(MouseScrollEvent &event) override;

  // Draw event
  void drawEvent() override;

  ImGuiIntegration::Context _imgui{NoCreate};
  GL::Mesh grid_mesh_{NoCreate}, origin_axis_mesh_{NoCreate};

  SceneGraph::Camera3D *camera_;

  Float _lastDepth;
  Vector2i last_position_{-1};
  Vector3 rotation_point_, translation_point_;
};

} // namespace Examples
} // namespace Magnum
