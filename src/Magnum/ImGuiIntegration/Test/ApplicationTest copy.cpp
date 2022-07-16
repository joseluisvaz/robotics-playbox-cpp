/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                2020, 2021, 2022 Vladimír Vondruš <mosra@centrum.cz>
    Copyright © 2018 ShaddyAQN <ShaddyAQN@gmail.com>
    Copyright © 2018 Tomáš Skřivan <skrivantomas@seznam.cz>
    Copyright © 2018 Jonathan Hale <squareys@googlemail.com>
    Copyright © 2018 Natesh Narain <nnaraindev@gmail.com>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*/

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Shaders/VertexColorGL.h>

#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Axis.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Shaders/PhongGL.h>
#include <Magnum/Trade/MeshData.h>

#ifdef CORRADE_TARGET_ANDROID
#include <Magnum/Platform/AndroidApplication.h>
#elif defined(CORRADE_TARGET_EMSCRIPTEN)
#include <Magnum/Platform/EmscriptenApplication.h>
#else
#include <Magnum/Platform/Sdl2Application.h>
#endif

namespace Magnum {
namespace ImGuiIntegration {
namespace Test {

using namespace Math::Literals;

class ApplicationTest : public Platform::Application {
public:
  explicit ApplicationTest(const Arguments &arguments);

  void drawEvent() override;

  void viewportEvent(ViewportEvent &event) override;

#ifndef CORRADE_TARGET_ANDROID
  void keyPressEvent(KeyEvent &event) override;
  void keyReleaseEvent(KeyEvent &event) override;
#endif

  void mousePressEvent(MouseEvent &event) override;
  void mouseReleaseEvent(MouseEvent &event) override;
  void mouseMoveEvent(MouseMoveEvent &event) override;
#ifndef CORRADE_TARGET_ANDROID
  void mouseScrollEvent(MouseScrollEvent &event) override;
  void textInputEvent(TextInputEvent &event) override;
#endif

private:
  ImGuiIntegration::Context _imgui{NoCreate};

  GL::Mesh _mesh;
  Shaders::PhongGL _shader;
  Matrix4 _transformation, _projection;
};

ApplicationTest::ApplicationTest(const Arguments &arguments)
    : Platform::Application{
          arguments, Configuration{}
                         .setTitle("Magnum ImGui Application Test")
#ifndef CORRADE_TARGET_ANDROID
                         .setWindowFlags(Configuration::WindowFlag::Resizable)
#endif
      } {

  _imgui = ImGuiIntegration::Context(Vector2{windowSize()} / dpiScaling(),
                                     windowSize(), framebufferSize());

  GL::Renderer::enable(GL::Renderer::Feature::Blending);
  GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);
  GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add,
                                 GL::Renderer::BlendEquation::Add);
  GL::Renderer::setBlendFunction(
      GL::Renderer::BlendFunction::SourceAlpha,
      GL::Renderer::BlendFunction::OneMinusSourceAlpha);

#if !defined(MAGNUM_TARGET_WEBGL) && !defined(CORRADE_TARGET_ANDROID)
  /* Have some sane speed, please */
  setMinimalLoopPeriod(16);
#endif

  _mesh = MeshTools::compile(Primitives::cubeWireframe());

  _transformation =
      Matrix4::rotationX(30.0_degf) * Matrix4::rotationY(40.0_degf);
  _projection =
      Matrix4::perspectiveProjection(
          35.0_degf, Vector2{windowSize()}.aspectRatio(), 0.01f, 100.0f) *
      Matrix4::translation(Vector3::zAxis(-10.0f));
}

void ApplicationTest::drawEvent() {
  GL::defaultFramebuffer.clear(GL::FramebufferClear::Color |
                               GL::FramebufferClear::Depth);

  {
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    auto axis_mesh = MeshTools::compile(Primitives::axis3D());
    _shader.setLightPositions({{0.5f, 1.0f, 0.75f, 0.0f}})
        .setTransformationMatrix(_transformation)
        .setNormalMatrix(_transformation.normalMatrix())
        .setProjectionMatrix(_projection)
        .draw(_mesh)
        .draw(axis_mesh);

    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
  }

  _imgui.newFrame();
  ImGui::ShowDemoWindow();
  _imgui.drawFrame();

  swapBuffers();
  redraw();
}

void ApplicationTest::viewportEvent(ViewportEvent &event) {
  GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});

  _imgui.relayout(Vector2{event.windowSize()} / event.dpiScaling(),
                  event.windowSize(), event.framebufferSize());
}

#ifndef CORRADE_TARGET_ANDROID
void ApplicationTest::keyPressEvent(KeyEvent &event) {
  if (_imgui.handleKeyPressEvent(event))
    return;
}

void ApplicationTest::keyReleaseEvent(KeyEvent &event) {
  if (_imgui.handleKeyReleaseEvent(event))
    return;
}
#endif

void ApplicationTest::mousePressEvent(MouseEvent &event) {
  if (_imgui.handleMousePressEvent(event))
    return;
}

void ApplicationTest::mouseReleaseEvent(MouseEvent &event) {
  if (_imgui.handleMouseReleaseEvent(event))
    return;
}

void ApplicationTest::mouseMoveEvent(MouseMoveEvent &event) {
  if (_imgui.handleMouseMoveEvent(event))
    return;

  if (!(event.buttons() & MouseMoveEvent::Button::Left))
    return;

  Vector2 delta =
      3.0f * Vector2{event.relativePosition()} / Vector2{windowSize()};

  _transformation = Matrix4::rotationX(Rad{delta.y()}) * _transformation *
                    Matrix4::rotationY(Rad{delta.x()});

  event.setAccepted();
  redraw();
}

#ifndef CORRADE_TARGET_ANDROID
void ApplicationTest::mouseScrollEvent(MouseScrollEvent &event) {
  if (_imgui.handleMouseScrollEvent(event)) {
    /* Prevent scrolling the page */
    event.setAccepted();
    return;
  }
}

void ApplicationTest::textInputEvent(TextInputEvent &event) {
  if (_imgui.handleTextInputEvent(event))
    return;
}
#endif

} // namespace Test
} // namespace ImGuiIntegration
} // namespace Magnum

MAGNUM_APPLICATION_MAIN(Magnum::ImGuiIntegration::Test::ApplicationTest)
