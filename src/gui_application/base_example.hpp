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

#include "cem_mpc.hpp"
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
using namespace RoboticsSandbox;

class BaseExample : public Platform::Application
{

public:
  using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
  using Scene3D = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

  explicit BaseExample(const Arguments &arguments);

  void resetCameraPosition();

  // functions to overwrite to create new applications/examples.
  virtual void show_menu() = 0;
  virtual void execute() = 0;

  Scene3D _scene;
  SceneGraph::DrawableGroup3D _drawables;

  // Shader to render meshes with position and color information
  Shaders::VertexColorGL3D _vertexColorShader{NoCreate};

  // Shader to render meshes with color information
  Shaders::MeshVisualizerGL3D _wireframe_shader{NoCreate};

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
  Shaders::FlatGL3D _flatShader{NoCreate};
  GL::Mesh _grid_mesh{NoCreate}, _origin_axis_mesh{NoCreate};

  Object3D *_cameraObject;
  SceneGraph::Camera3D *_camera;

  Float _lastDepth;
  Vector2i _lastPosition{-1};
  Vector3 _rotationPoint, _translationPoint;
};

} // namespace Examples
} // namespace Magnum
