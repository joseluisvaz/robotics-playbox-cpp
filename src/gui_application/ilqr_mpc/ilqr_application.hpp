
// Copyright © 2022 <copyright holders>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the “Software”), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE

#include <algorithm>
#include <optional>

#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Scene.h>

#include "base_application/base_application.hpp"
#include "common/containers.h"
#include "common/types.hpp"
#include "environment/lane_map.hpp"
#include "graphics/graphics_objects.hpp"
#include "ilqr_mpc/ilqr_mpc.hpp"

namespace mpex
{

using namespace Eigen;

class IlqrMain : public Magnum::Examples::BaseApplication
{

public:
  using Dynamics = EigenKinematicBicycle;
  explicit IlqrMain(const Arguments &arguments);

private:
  virtual void show_menu() final;
  virtual void execute() final;
  void run_ilqr();
  Magnum::GL::Mesh mesh_{Magnum::NoCreate};
  Graphics::TrajectoryEntities trajectory_entities_;
  IterativeLinearQuadraticRegulator policy_;
  std::optional<Dynamics::Trajectory> maybe_current_trajectory_;
  bool is_running_{false};

  // State
  Dynamics::State current_state_;
  double time_;

  // enviroment
  environment::Lane lane_;
  Graphics::LaneEntity lane_entity_;

  std::unordered_map<std::string, containers::Buffer<double>> history_buffer_;
};

} // namespace mpex
