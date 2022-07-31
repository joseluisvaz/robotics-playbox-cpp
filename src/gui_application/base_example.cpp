#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Image.h>
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

#include "gui_application/graphics_objects.hpp"
#include "gui_application/implot.h"
#include "gui_application/types.hpp"
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <algorithm>
#include <easy/profiler.h>
#include <iostream>
#include <memory>

#include "gui_application/gui_application.hh"

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
  new FlatDrawable{*grid, _flatShader, _grid, _drawables};

  /* Origin Axis */
  _origin_axis = MeshTools::compile(Primitives::axis3D());
  auto origin_axis = new Object3D(&_scene);
  new VertexColorDrawable{*origin_axis, _vertexColorShader, _origin_axis, _drawables};

  /* Set up the camera */
  _cameraObject = new Object3D{&_scene};
  this->resetCameraPosition();

  _camera = new SceneGraph::Camera3D{*_cameraObject};
  _camera->setProjectionMatrix(
      Matrix4::perspectiveProjection(45.0_degf, Vector2{windowSize()}.aspectRatio(), 0.01f, 100.0f));

  /* Initialize initial depth to the value at scene center */
  _lastDepth = ((_camera->projectionMatrix() * _camera->cameraMatrix()).transformPoint({}).z() + 1.0f) * 0.5f;

  _mesh = MeshTools::compile(Primitives::cubeWireframe());
  constexpr int horizon = 20;
  constexpr int population = 1024;
  mpc_ = CEM_MPC<EigenKinematicBicycle>(/* iters= */ 20, horizon,
                                        /* population= */ population,
                                        /* elites */ 16);

  trajectory_objects_ = Graphics::TrajectoryObjects(_scene, horizon);
  for (auto &object : trajectory_objects_.get_objects())
  {
    new VertexColorDrawable{*object, _vertexColorShader, _mesh, _drawables};
  }

  for (int i = 0; i < population; ++i)
  {
    auto path_object = std::make_shared<Graphics::PathObjects>();
    path_objects_.push_back(path_object);
    auto vtx = new Object3D{&_scene};
    new VertexColorDrawable{*vtx, _vertexColorShader, path_object->mesh_, _drawables};
  }

  this->runCEM();
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
  // ImGui::ShowDemoWindow();

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

void BaseExample::execute()
{
  if (is_running_)
  {
    runCEM();
  }
}

void BaseExample::show_menu()
{
  EASY_FUNCTION(profiler::colors::Blue);
  ImGui::SetNextWindowPos({500.0f, 50.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.5f);
  ImGui::Begin("Options", nullptr);

  if (ImGui::Button("Reset scene"))
  {
    resetCameraPosition();
    redraw();
  }

  if (ImGui::Button("Run MPC"))
  {
    is_running_ = is_running_ ? false : true;
    redraw();
  }

  ImGui::SliderInt("Num Iterations", &mpc_.get_num_iters_mutable(), 1, 100);
  ImGui::SliderFloat3("costs[x, y, yaw]", mpc_.cost_function_.state_slider_values_, 0.0f, 10.0f);
  ImGui::SliderFloat3("costs[speed, acc, steering]", mpc_.cost_function_.state_slider_values_2_, 0.0f, 10.0f);
  ImGui::SliderFloat2("costs[jerk, steering]", mpc_.cost_function_.action_slider_values_, 0.0f, 10.0f);
  ImGui::SliderFloat3("ref[x, y, speed]", mpc_.cost_function_.ref_values_, 0.0f, 10.0f);
  ImGui::SliderAngle("ref[]", &mpc_.cost_function_.ref_yaw_, 0.0f, 180.0f);

  ImGui::SliderFloat3("terminal costs[x, y, yaw]", mpc_.cost_function_.terminal_state_slider_values_, 0.0f, 10.0f);
  ImGui::SliderFloat3("terminal costs[speed, acc, steering]", mpc_.cost_function_.terminal_state_slider_values_2_, 0.0f,
                      10.0f);
  ImGui::SliderFloat2("terminal costs[jerk, steering]", mpc_.cost_function_.terminal_action_slider_values_, 0.0f,
                      10.0f);
  ImGui::SliderFloat3("terminal ref[x, y, speed]", mpc_.cost_function_.terminal_ref_values_, -100.0f, 100.0f);
  ImGui::SliderAngle("terminal ref[]", &mpc_.cost_function_.terminal_ref_yaw_, -180.0f, 180.0f);

  const auto make_plot = [this](auto title, auto value_name, auto getter_fn)
  {
    ImPlot::SetNextAxesToFit();
    if (ImPlot::BeginPlot(title))
    {
      ImPlot::SetupAxes("time[s]", value_name);

      for (auto traj : this->mpc_.candidate_trajectories_)
      {
        std::vector<float> values;
        std::transform(traj.states.colwise().begin(), traj.states.colwise().end(), std::back_inserter(values),
                       getter_fn);
        ImPlot::PlotLine(value_name, traj.times.data(), values.data(), values.size());
      }

      std::vector<float> values;
      std::transform(this->_trajectory.states.colwise().begin(), this->_trajectory.states.colwise().end(),
                     std::back_inserter(values), getter_fn);

      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
      ImPlot::PlotLine(std::string(value_name).append("_hola").c_str(), this->_trajectory.times.data(), values.data(),
                       values.size());
      ImPlot::PopStyleColor();

      ImPlot::EndPlot();
    }
  };

  const auto make_action_plot = [this](auto title, auto value_name, auto getter_fn)
  {
    ImPlot::SetNextAxesToFit();
    if (ImPlot::BeginPlot(title))
    {
      ImPlot::SetupAxes("time[s]", value_name);
      for (auto traj : this->mpc_.candidate_trajectories_)
      {
        std::vector<float> values;
        std::transform(traj.actions.colwise().begin(), traj.actions.colwise().end(), std::back_inserter(values),
                       getter_fn);
        ImPlot::PlotLine(value_name, traj.times.data(), values.data(), values.size());
      }

      std::vector<float> values;
      std::transform(this->_trajectory.actions.colwise().begin(), this->_trajectory.actions.colwise().end(),
                     std::back_inserter(values), getter_fn);

      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
      ImPlot::PlotLine(std::string(value_name).append("_hola").c_str(), this->_trajectory.times.data(), values.data(),
                       values.size());
      ImPlot::PopStyleColor();
      ImPlot::EndPlot();
    }
  };

  EASY_BLOCK("Make plots");
  make_plot("Speed Plot", "speed[mps]", [](const Ref<typename Dynamics::State> &state) { return state[3]; });
  make_plot("Accel Plot", "accel[mpss]", [](const Ref<typename Dynamics::State> &state) { return state[4]; });
  make_plot("Yaw Plot", "yaw[rad]", [](const Ref<typename Dynamics::State> &state) { return state[2]; });
  make_plot("Steering Plot", "steering[rad]", [](const Ref<typename Dynamics::State> &state) { return state[5]; });
  make_action_plot("Jerk Plot", "jerk[mpsss]",
                   [](const Ref<typename Dynamics::Action> &action) { return 0.6 * std::tanh(action[0]); });
  make_action_plot("Steering Rate Plot", "srate[rad/s]",
                   [](const Ref<typename Dynamics::Action> &action) { return 0.1 * std::tanh(action[1]); });
  EASY_END_BLOCK;

  ImGui::End();
}

void BaseExample::resetCameraPosition()
{
  if (!_cameraObject)
  {
    return;
  }
  (*_cameraObject)
      .resetTransformation()
      .translate(Vector3::zAxis(SCALE(100.0f)))
      .rotateX(-15.0_degf)
      .rotateY(30.0_degf);
}

void BaseExample::runCEM()
{
  EASY_FUNCTION(profiler::colors::Red);

  Dynamics::State state = Dynamics::State::Zero();
  state[3] = 10.0;
  Dynamics::Trajectory &trajectory = mpc_.execute(state);
  _trajectory = trajectory;

  EASY_BLOCK("Plotting All");
  for (int i = 0; i < trajectory.states.cols(); ++i)
  {
    Dynamics::State new_state = trajectory.states.col(i);
    auto time_s = trajectory.times.at(i);
    auto &object = trajectory_objects_.get_objects().at(i);

    EASY_BLOCK("Plotting");
    (*object)
        .resetTransformation()
        .scale(trajectory_objects_.get_vehicle_extent())
        .translate(Vector3(0.0f, 0.0f, SCALE(1.5f))) // move half wheelbase forward
        .rotateY(Rad(new_state[2]))
        .translate(Vector3(SCALE(new_state[1]), SCALE(time_s), SCALE(new_state[0])));
    EASY_END_BLOCK;
  }

  const auto set_path_helper = [this](const auto &_trajectory, auto &_path_object)
  {
    std::vector<float> x_vals;
    std::vector<float> y_vals;
    std::vector<float> t_vals;
    for (int i = 0; i < _trajectory.states.cols(); ++i)
    {
      Dynamics::State new_state = _trajectory.states.col(i);
      auto time_s = _trajectory.times.at(i);

      x_vals.push_back(new_state[0]);
      y_vals.push_back(new_state[1]);
      t_vals.push_back(time_s);
    }

    _path_object.set_path(x_vals, y_vals, t_vals);
  };

  assert(path_objects_.size() == mpc_.candidate_trajectories_.size());
  for (int i = 0; i < path_objects_.size(); ++i)
  {
    set_path_helper(mpc_.candidate_trajectories_.at(i), *path_objects_.at(i));
  }

  EASY_END_BLOCK;
}

} // namespace Examples
} // namespace Magnum

MAGNUM_APPLICATION_MAIN(Magnum::Examples::BaseExample)