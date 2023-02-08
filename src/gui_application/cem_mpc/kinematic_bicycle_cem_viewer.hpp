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

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Scene.h>

#include "cem_mpc/cem_mpc.h"
#include "common/dynamics.h"
#include "graphics/graphics_objects.hpp"

namespace mpex {

namespace impl {

template <typename Dynamics>
void draw_candidate_paths(
    CEM_MPC<Dynamics> &mpc,
    std::vector<std::shared_ptr<Graphics::LineEntity>> &path_entities,
    std::shared_ptr<Graphics::LineEntity> &optimal_path_entity)
{
    const auto set_xy_helper = [&](const auto &candidate_trajectory, auto &path_entity, auto color) {
        std::vector<double> x_vals;
        std::vector<double> y_vals;
        std::vector<double> z_vals;
        for (int i = 0; i < candidate_trajectory.states.cols(); ++i)
        {
            const typename Dynamics::State state = candidate_trajectory.states.col(i);
            auto se2 = mpc.dynamics_ptr_->get_track_ptr()->convert_to_se2(state[0], state[1], state[2]);
            x_vals.push_back(se2.x);
            y_vals.push_back(se2.y);
            z_vals.push_back(state[3]);
        }
        path_entity.set_xy(x_vals.size(), x_vals.data(), y_vals.data(), z_vals.data(), color);
    };

    const auto max_iter = std::max_element(mpc.costs_index_pair_.begin(), mpc.costs_index_pair_.end());
    const auto min_iter = std::min_element(mpc.costs_index_pair_.begin(), mpc.costs_index_pair_.end());
    const auto max_cost = max_iter != mpc.costs_index_pair_.end() ? max_iter->first : 1e9;
    const auto min_cost = min_iter != mpc.costs_index_pair_.end() ? min_iter->first : -1e9;

    assert(path_entities.size() == mpc.candidate_trajectories_.size());
    for (int i = 0; i < path_entities.size(); ++i)
    {
        float cost = mpc.costs_index_pair_[i].first;
        cost = (cost - min_cost) / (max_cost - min_cost); // normalize

        const auto exp_color_value_blue = std::exp(-10.0f * cost);
        const auto exp_color_value_red = 1.0f - exp_color_value_blue;
        const auto color = Magnum::Math::Color3(exp_color_value_red, 0.0f, exp_color_value_blue);

        auto &candidate_traj = mpc.candidate_trajectories_[i];
        set_xy_helper(candidate_traj, *path_entities.at(i), color);
    }

    const auto &optimal_trajectory = mpc.get_last_solution();
    const auto color = Magnum::Math::Color3(0.5f, 1.0f, 0.5f);
    set_xy_helper(optimal_trajectory, *optimal_path_entity, color);
}

} // namespace impl

using Object3D = Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>;
using Scene3D = Magnum::SceneGraph::Scene<Magnum::SceneGraph::MatrixTransformation3D>;

const auto kRedColor = Magnum::Math::Color3(1.0f, 0.2f, 0.0f);

template <typename Dynamics = EigenKinematicBicycle>
class KinematicBicycleCemViewer
{

  public:
    KinematicBicycleCemViewer() = default;

    KinematicBicycleCemViewer(
        std::shared_ptr<CEM_MPC<Dynamics>> mpc_ptr,
        Scene3D &scene,
        Magnum::Shaders::MeshVisualizerGL3D &shader,
        Magnum::Shaders::FlatGL3D &flat_shader,
        Magnum::SceneGraph::DrawableGroup3D &group,
        int horizon,
        int population)
        : mpc_ptr_(mpc_ptr)
    {
        for (int i = 0; i < population; ++i)
        {
            path_entities_.push_back(std::make_shared<Graphics::LineEntity>(scene));
            new Graphics::FlatDrawable{*path_entities_.back()->object_ptr_, flat_shader, path_entities_.back()->mesh_, group};
        }

        optimal_path_entity_ = std::make_shared<Graphics::LineEntity>(scene);
        new Graphics::FlatDrawable{*optimal_path_entity_->object_ptr_, flat_shader, optimal_path_entity_->mesh_, group};

        trajectory_entities_ = std::make_shared<Graphics::TrajectoryEntities>(scene, horizon);
        for (auto &object : trajectory_entities_->get_objects())
        {
            new Graphics::WireframeDrawable{*object, shader, trajectory_entities_->mesh_, group, kRedColor};
        }
    };

    void draw()
    {
        impl::draw_candidate_paths<Dynamics>(*mpc_ptr_, path_entities_, optimal_path_entity_);

        // Draw bounding boxes of trajectory.
        auto trajectory = mpc_ptr_->get_last_solution();
        for (int i = 0; i < trajectory.states.cols(); ++i)
        {
            typename Dynamics::State state = trajectory.states.col(i);

            // TODO: Create an SE2 utility function
            auto se2 = mpc_ptr_->dynamics_ptr_->get_track_ptr()->convert_to_se2(state[0], state[1], state[2]);
            trajectory_entities_->set_state_at(i, se2.x, se2.y, se2.theta);
        }
    }

    std::shared_ptr<Graphics::TrajectoryEntities> trajectory_entities_;
    std::vector<std::shared_ptr<Graphics::LineEntity>> path_entities_;
    std::shared_ptr<Graphics::LineEntity> optimal_path_entity_;
    std::shared_ptr<CEM_MPC<Dynamics>> mpc_ptr_;
};

} // namespace mpex
