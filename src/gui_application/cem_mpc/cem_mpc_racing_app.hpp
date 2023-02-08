
#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Magnum/GL/Mesh.h>
#include <Magnum/Platform/Sdl2Application.h>

#include "base_application/base_application.hpp"
#include "cem_mpc/cem_mpc.h"
#include "cem_mpc/intelligent_driver_model.hpp"
#include "cem_mpc/kinematic_bicycle_cem_viewer.hpp"
#include "common/containers.h"
#include "common/dynamics.h"
#include "common/types.hpp"
#include "environment/lane_map.hpp"
#include "environment/track.hpp"
#include "graphics/graphics_objects.hpp"

namespace mpex {

using namespace Eigen;

class CEMMPCRacingApp : public Magnum::Examples::BaseApplication
{
    // using Dynamics = EigenKinematicBicycle;
    using Dynamics = CurvilinearKinematicBicycle;

  public:
    explicit CEMMPCRacingApp(const Arguments &arguments);

  private:
    virtual void show_menu();
    virtual void execute();

    void runCEM();

    // Visualization entities for plotting using Magnum
    KinematicBicycleCemViewer<Dynamics> mpc_viewer_;
    std::shared_ptr<Graphics::LineEntity> left_boundary_;
    std::shared_ptr<Graphics::LineEntity> centerline_;
    std::shared_ptr<Graphics::LineEntity> right_boundary_;

    // Flags to control the GUI
    bool is_running_{false};

    // Ego policy and state information, used to control the agent that is using the MPC
    std::shared_ptr<CEM_MPC<Dynamics>> ego_policy_ptr_;
    Dynamics::State ego_state_;

    // Pointer to the dynamics object
    std::shared_ptr<Dynamics> dynamics_ptr_;

    // Track used by the MPC
    std::shared_ptr<environment::Track> track_ptr_;

    // State of the simulation in this case the time_s
    double time_s_;
    std::unordered_map<std::string, containers::Buffer<double>> history_buffer_;
    std::optional<Dynamics::Trajectory> maybe_current_trajectory_;
};

} // namespace mpex
