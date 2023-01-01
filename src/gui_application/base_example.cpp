#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Image.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/FunctionsBatch.h>
#include <Magnum/MeshTools/Compile.h>
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
#include <Magnum/ImGuiIntegration/Context.hpp>

#include "gui_application/graphics/graphics_objects.hpp"
#include "gui_application/implot.h"
#include "gui_application/types.hpp"
#include <algorithm>
#include <easy/profiler.h>
#include <iostream>
#include <memory>

#include "gui_application/base_example.hpp"

namespace Magnum
{
namespace Examples
{

BaseExample::BaseExample(const Arguments &arguments) : Platform::Application{arguments, NoCreate}
{
  profiler::startListen();
  /* Setup window */
  {
    const Vector2 dpiScaling = this->dpiScaling({});
    Configuration conf;
    conf.setTitle("Robotics Sandbox")
        .setSize(conf.size(), dpiScaling)
        .setWindowFlags(Configuration::WindowFlag::Resizable | Configuration::WindowFlag::Maximized |
                        Configuration::WindowFlag::Tooltip);
    GLConfiguration glConf;
    glConf.setSampleCount(dpiScaling.max() < 2.0f ? 8 : 2);
    if (!tryCreate(conf, glConf))
    {
      create(conf, glConf.setSampleCount(0));
    }
  }

  /* Setup ImGui, load a better font */
  {
    _imgui = ImGuiIntegration::Context(Vector2{windowSize()} / dpiScaling(), windowSize(), framebufferSize());
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();

    /* Setup proper blending to be used by ImGui */
    GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add, GL::Renderer::BlendEquation::Add);
    GL::Renderer::setBlendFunction(GL::Renderer::BlendFunction::SourceAlpha,
                                   GL::Renderer::BlendFunction::OneMinusSourceAlpha);
    GL::Renderer::setLineWidth(3.0f);
  }

  /* Shaders, renderer setup */
  _vertexColorShader = Shaders::VertexColorGL3D{};
  _flatShader = Shaders::FlatGL3D{};
  GL::Renderer::enable(GL::Renderer::Feature::DepthTest);

  /* Grid */
  _grid = MeshTools::compile(Primitives::grid3DWireframe({15, 15}));
  auto grid = new Object3D{&_scene};
  (*grid).rotateX(90.0_degf).scale(Vector3{15.0f});
  new Graphics::FlatDrawable{*grid, _flatShader, _grid, _drawables};

  /* Origin Axis */
  _origin_axis = MeshTools::compile(Primitives::axis3D());
  auto origin_axis = new Object3D(&_scene);
  new Graphics::VertexColorDrawable{*origin_axis, _vertexColorShader, _origin_axis, _drawables};

  /* Set up the camera */
  _cameraObject = new Object3D{&_scene};
  this->resetCameraPosition();

  _camera = new SceneGraph::Camera3D{*_cameraObject};
  _camera->setProjectionMatrix(
      Matrix4::perspectiveProjection(45.0_degf, Vector2{windowSize()}.aspectRatio(), 0.01f, 100.0f));

  /* Initialize initial depth to the value at scene center */
  _lastDepth = ((_camera->projectionMatrix() * _camera->cameraMatrix()).transformPoint({}).z() + 1.0f) * 0.5f;
}

Float BaseExample::depthAt(const Vector2i &windowPosition)
{
  /* First scale the position from being relative to window size to being
     relative to framebuffer size as those two can be different on HiDPI
     systems */
  const Vector2i position = windowPosition * Vector2{framebufferSize()} / Vector2{windowSize()};
  const Vector2i fbPosition{position.x(), GL::defaultFramebuffer.viewport().sizeY() - position.y() - 1};

  GL::defaultFramebuffer.mapForRead(GL::DefaultFramebuffer::ReadAttachment::Front);
  Image2D data = GL::defaultFramebuffer.read(Range2Di::fromSize(fbPosition, Vector2i{1}).padded(Vector2i{2}),
                                             {GL::PixelFormat::DepthComponent, GL::PixelType::Float});

  /* TODO: change to just Math::min<Float>(data.pixels<Float>() when the
     batch functions in Math can handle 2D views */
  return Math::min<Float>(data.pixels<Float>().asContiguous());
}

Vector3 BaseExample::unproject(const Vector2i &windowPosition, Float depth) const
{
  /* We have to take window size, not framebuffer size, since the position is
     in window coordinates and the two can be different on HiDPI systems */
  const Vector2i viewSize = windowSize();
  const Vector2i viewPosition{windowPosition.x(), viewSize.y() - windowPosition.y() - 1};
  const Vector3 in{2 * Vector2{viewPosition} / Vector2{viewSize} - Vector2{1.0f}, depth * 2.0f - 1.0f};

  /*
      Use the following to get global coordinates instead of
     camera-relative:
      (_cameraObject->absoluteTransformationMatrix()*_camera->projectionMatrix().inverted()).transformPoint(in)
  */
  return _camera->projectionMatrix().inverted().transformPoint(in);
}

void BaseExample::viewportEvent(ViewportEvent &event)
{
  /* Resize the main framebuffer */
  GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});

  /* Relayout ImGui */
  _imgui.relayout(Vector2{event.windowSize()} / event.dpiScaling(), event.windowSize(), event.framebufferSize());

  /* Recompute the camera's projection matrix */
  _camera->setViewport(event.framebufferSize());
}

void BaseExample::keyPressEvent(KeyEvent &event)
{
  if (_imgui.handleKeyPressEvent(event))
    return;

  /* Reset the transformation to the original view */
  if (event.key() == KeyEvent::Key::NumZero)
  {
    (*_cameraObject).resetTransformation().translate(Vector3::zAxis(5.0f)).rotateX(-15.0_degf).rotateY(30.0_degf);
    redraw();
    return;

    /* Axis-aligned view */
  }
  else if (event.key() == KeyEvent::Key::NumOne || event.key() == KeyEvent::Key::NumThree ||
           event.key() == KeyEvent::Key::NumSeven)
  {
    /* Start with current camera translation with the rotation inverted */
    const Vector3 viewTranslation =
        _cameraObject->transformation().rotationScaling().inverted() * _cameraObject->transformation().translation();

    /* Front/back */
    const Float multiplier = event.modifiers() & KeyEvent::Modifier::Ctrl ? -1.0f : 1.0f;

    Matrix4 transformation;
    if (event.key() == KeyEvent::Key::NumSeven) /* Top/bottom */
      transformation = Matrix4::rotationX(-90.0_degf * multiplier);
    else if (event.key() == KeyEvent::Key::NumOne) /* Front/back */
      transformation = Matrix4::rotationY(90.0_degf - 90.0_degf * multiplier);
    else if (event.key() == KeyEvent::Key::NumThree) /* Right/left */
      transformation = Matrix4::rotationY(90.0_degf * multiplier);
    else
      CORRADE_INTERNAL_ASSERT_UNREACHABLE();

    _cameraObject->setTransformation(transformation * Matrix4::translation(viewTranslation));
    redraw();
  }
}

void BaseExample::mousePressEvent(MouseEvent &event)
{
  if (_imgui.handleMousePressEvent(event))
    return;

  /* Due to compatibility reasons, scroll is also reported as a press event,
     so filter that out. Could be removed once MouseEvent::Button::Wheel is
     gone from Magnum. */
  if (event.button() != MouseEvent::Button::Left && event.button() != MouseEvent::Button::Middle)
    return;

  const Float currentDepth = depthAt(event.position());
  const Float depth = currentDepth == 1.0f ? _lastDepth : currentDepth;
  _translationPoint = unproject(event.position(), depth);
  /* Update the rotation point only if we're not zooming against infinite
     depth or if the original rotation point is not yet initialized */
  if (currentDepth != 1.0f || _rotationPoint.isZero())
  {
    _rotationPoint = _translationPoint;
    _lastDepth = depth;
  }
}

void BaseExample::mouseMoveEvent(MouseMoveEvent &event)
{
  if (_imgui.handleMouseMoveEvent(event))
    return;

  if (_lastPosition == Vector2i{-1})
    _lastPosition = event.position();
  const Vector2i delta = event.position() - _lastPosition;
  _lastPosition = event.position();

  if (!event.buttons())
    return;

  /* Translate */
  if (event.modifiers() & MouseMoveEvent::Modifier::Shift)
  {
    const Vector3 p = unproject(event.position(), _lastDepth);
    _cameraObject->translateLocal(_translationPoint - p); /* is Z always 0? */
    _translationPoint = p;

    /* Rotate around rotation point */
  }
  else
  {
    _cameraObject->transformLocal(Matrix4::translation(_rotationPoint) * Matrix4::rotationX(-0.01_radf * delta.y()) *
                                  Matrix4::rotationY(-0.01_radf * delta.x()) * Matrix4::translation(-_rotationPoint));
  }

  redraw();
}

void BaseExample::mouseScrollEvent(MouseScrollEvent &event)
{
  if (_imgui.handleMouseScrollEvent(event))
  {
    /* Prevent scrolling the page */
    event.setAccepted();
    return;
  }

  const Float currentDepth = depthAt(event.position());
  const Float depth = currentDepth == 1.0f ? _lastDepth : currentDepth;
  const Vector3 p = unproject(event.position(), depth);
  /* Update the rotation point only if we're not zooming against infinite
     depth or if the original rotation point is not yet initialized */
  if (currentDepth != 1.0f || _rotationPoint.isZero())
  {
    _rotationPoint = p;
    _lastDepth = depth;
  }

  const Float direction = event.offset().y();
  if (!direction)
    return;

  /* Move towards/backwards the rotation point in cam coords */
  _cameraObject->translateLocal(_rotationPoint * direction * 0.1f);

  event.setAccepted();
  redraw();
}

void BaseExample::mouseReleaseEvent(MouseEvent &event)
{
  if (_imgui.handleMouseReleaseEvent(event))
    return;
}

void BaseExample::drawEvent()
{
  GL::defaultFramebuffer.clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);
  _imgui.newFrame();

  /* Enable text input, if needed */
  if (ImGui::GetIO().WantTextInput && !isTextInputActive())
  {
    startTextInput();
  }
  else if (!ImGui::GetIO().WantTextInput && isTextInputActive())
  {
    stopTextInput();
  }

  // Execute the main content of the application, data generation, adding to drawables etc.
  execute();

  _camera->draw(_drawables);

  // Set Imgui drawables
  show_menu();

  /* Update application cursor */
  _imgui.updateApplicationCursor(*this);

  /* Render ImGui window */
  {
    GL::Renderer::enable(GL::Renderer::Feature::Blending);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);

    _imgui.drawFrame();

    GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::Blending);
  }
  swapBuffers();
  /* Run next frame immediately */
  redraw();
}

void BaseExample::resetCameraPosition()
{
  if (!_cameraObject)
  {
    return;
  }
  (*_cameraObject)
      .resetTransformation()
      .translate(Vector3::zAxis(SCALE(200.0f)))
      .translate(Vector3::xAxis(SCALE(50.0f)))
      .rotateX(-90.0_degf)
      .rotateY(-90.0_degf); 
}

} // namespace Examples
} // namespace Magnum

// MAGNUM_APPLICATION_MAIN(Magnum::Examples::BaseExample)