#pragma once

#include <Corrade/Containers/ArrayViewStl.h>
#include <Magnum/GL/Buffer.h>
// #include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
// #include <Magnum/GL/PixelFormat.h>
// #include <Magnum/GL/Renderer.h>
// #include <Magnum/Image.h>
#include <Magnum/Math/Color.h>
// #include <Magnum/Math/FunctionsBatch.h>
// #include <Magnum/MeshTools/Compile.h>
// #include <Magnum/Platform/Sdl2Application.h>
// #include <Magnum/Primitives/Axis.h>
// #include <Magnum/Primitives/Cube.h>
// #include <Magnum/Primitives/Grid.h>
// #include <Magnum/SceneGraph/Camera.h>
// #include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/Shaders/VertexColorGL.h>
#include <Magnum/Trade/MeshData.h>

#include <memory>
#include <vector>

#ifndef SCALE
#define SCALE(a) (a / 10.0)
#endif

namespace RoboticsSandbox::Graphics
{

using namespace Magnum;
using namespace Math::Literals;
using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
using Scene3D = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

class TrajectoryObjects
{
public:
  TrajectoryObjects() = default;
//   TrajectoryObjects(const TrajectoryObjects &other) = default;
//   TrajectoryObjects(TrajectoryObjects &&other) = default;
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
  PathObjects();

  void set_path(std::vector<float> x, std::vector<float> y, std::vector<float> z,
                const Color3 &color = Color3(1.0f, 1.0f, 0.0f));

  std::vector<Vertex> data_;
  GL::Buffer buffer_;
  GL::Mesh mesh_;
  Shaders::VertexColorGL3D shader_;
};

} // namespace RoboticsSandbox::Graphics
