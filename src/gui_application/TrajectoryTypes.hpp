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

#include <Eigen/Dense>
#include <array>
#include <easy/profiler.h>
#include <random>

namespace Magnum::Examples
{

using namespace Eigen;

constexpr int state_size = 6;
constexpr int action_size = 2;
using EigenState = Vector<float, state_size>;
using EigenAction = Vector<float, action_size>;

using EigenStateSequence = Matrix<float, state_size, Dynamic>;
using EigenActionSequence = Matrix<float, action_size, Dynamic>;

struct EigenTrajectory
{
  std::vector<float> times{};
  EigenStateSequence states;
  EigenActionSequence actions;
};

struct EigenKinematicBicycle
{
  constexpr static float ts{0.1};
  constexpr static float one_over_wheelbase{1.0 / 3.0};
  constexpr static float steering_max{0.52};

  static void step(const Ref<EigenState> &state, const Ref<EigenAction> &action,
                   Ref<EigenState> new_state)
  {
    new_state[0] = state[0] + ts * state[3] * std::cos(state[2]);
    new_state[1] = state[1] + ts * state[3] * std::sin(state[2]);
    new_state[2] =
        state[2] + ts * state[3] * one_over_wheelbase * std::tan(state[5]);
    new_state[3] = state[3] + ts * state[4];
    new_state[4] = state[4] + ts * action[0];
    new_state[5] = state[5] + ts * action[1];
  }
};

struct NormalRandomVariable
{
  NormalRandomVariable() = default;
  NormalRandomVariable(const EigenActionSequence &mean,
                       const EigenActionSequence &covar)
      : mean(mean), transform(covar){};

  EigenActionSequence mean;
  EigenActionSequence transform;

  EigenActionSequence operator()() const
  {
    static std::mt19937 gen{std::random_device{}()};
    static std::normal_distribution<float> dist;

    return mean.array() +
           transform.array() *
               (EigenActionSequence::Zero(mean.rows(), mean.cols())
                    .unaryExpr([&](auto x) { return dist(gen); })
                    .array());
  }
};

template <typename DynamicsT>
class CEM_MPC
{

public:
  CEM_MPC() = default;
  CEM_MPC(const int num_iters, const int horizon, const int population)
      : num_iters_(num_iters), horizon_(horizon), population_(population)
  {
    trajectory_.states.conservativeResize(state_size, horizon_);
    trajectory_.actions.conservativeResize(action_size, horizon_);
    trajectory_.states = MatrixXf::Zero(state_size, horizon_);
    trajectory_.actions = MatrixXf::Zero(action_size, horizon_);
    trajectory_.times = std::vector<float>(horizon_, 0.0f);

    for (int i = 0; i < population_; ++i)
    {
      EigenTrajectory trajectory;
      trajectory.states.conservativeResize(state_size, horizon_);
      trajectory.actions.conservativeResize(action_size, horizon_);
      trajectory.states = MatrixXf::Zero(state_size, horizon_);
      trajectory.actions = MatrixXf::Zero(action_size, horizon_);
      trajectory.times = std::vector<float>(horizon_, 0.0f);
      candidate_trajectories_.emplace_back(std::move(trajectory));
    }

    EigenActionSequence mean = EigenActionSequence::Zero(
        trajectory_.actions.rows(), trajectory_.actions.cols());
    EigenActionSequence stddev = EigenActionSequence::Ones(
        trajectory_.actions.rows(), trajectory_.actions.cols());

    sampler_ = NormalRandomVariable(mean, stddev);
  };

  EigenTrajectory &rollout(const Ref<EigenState> &initial_state)
  {
    EASY_FUNCTION(profiler::colors::Magenta);

    EigenState state = initial_state;

    for (auto &trajectory : candidate_trajectories_)
    {
      EASY_BLOCK("Sampling");
      trajectory.actions = sampler_();
      EASY_END_BLOCK;

      double current_time_s{0.0};
      trajectory.states.col(0) = state; // Set first state
      for (size_t i = 0; i + 1 < horizon_; ++i)
      {
        trajectory.times.at(i) = current_time_s;

        EASY_BLOCK("Calculating dynamics");
        DynamicsT::step(trajectory.states.col(i), trajectory.actions.col(i),
                        trajectory.states.col(i + 1));
        EASY_END_BLOCK;

        current_time_s += DynamicsT::ts;
      }

      // Add last time
      trajectory.times.at(horizon_ - 1) = current_time_s;
    }

    return candidate_trajectories_.at(0);
  };

private:
  int num_iters_;
  int horizon_;
  int population_;
  std::vector<EigenTrajectory> candidate_trajectories_;
  EigenTrajectory trajectory_;
  NormalRandomVariable sampler_;
};

// struct KinematicBicycleState
// {
//   double x{0.0};
//   double y{0.0};
//   double yaw{0.0};
//   double speed{0.0};
//   double accel{0.0};
//   double steering{0.0};
// };

// struct KinematicBicycleAction
// {
//   double jerk{0.0};
//   double steering_rate{0.0};
// };

// template <typename StateT>
// struct Trajectory
// {
//   using StateType = StateT;

//   std::vector<double> time{};
//   std::vector<StateType> states{};
// };

// template <typename StateT, typename ActionT>
// struct StateActionTrajectory
// {
//   using StateType = StateT;
//   using ActionType = ActionT;

//   std::vector<double> time{};
//   std::vector<StateType> states{};
//   std::vector<ActionType> actions{};
// };

// using KinematicBicycleTrajectory = Trajectory<KinematicBicycleState>;

// struct KinematicBicycle
// {
//   constexpr static double ts{0.1};
//   constexpr static double one_over_wheelbase{1.0 / 3.0};
//   constexpr static double steering_max{0.52};

//   using StateType = KinematicBicycleState;
//   using ActionType = KinematicBicycleAction;

//   static StateType step(const StateType &state, const ActionType &action)
//   {
//     StateType new_state;

//     new_state.x = state.x + ts * state.speed * std::cos(state.yaw);
//     new_state.y = state.y + ts * state.speed * std::sin(state.yaw);
//     new_state.yaw =
//         state.yaw +
//         ts * state.speed * one_over_wheelbase *
//             std::tan(std::clamp(state.steering, -steering_max,
//             steering_max));
//     new_state.speed = state.speed + ts * state.accel;
//     new_state.accel = state.accel + ts * action.jerk;
//     new_state.steering = state.steering + ts * action.steering_rate;
//     return new_state;
//   }
// };

// template <typename DynamicsT>
// class CEM_MPC
// {
//   using StateType = typename DynamicsT::StateType;
//   using ActionType = typename DynamicsT::ActionType;

// public:
//   CEM_MPC() = default;
//   CEM_MPC(const int num_iters, const int horizon, const int population)
//       : num_iters_(num_iters), horizon_(horizon), population_(population){};

//   Trajectory<StateType> rollout(StateType initial_state)
//   {
//     std::random_device rd{};
//     std::mt19937 gen{rd()};

//     // values near the mean are the most likely
//     // standard deviation affects the dispersion of generated values from the
//     // mean
//     std::normal_distribution<> d{0, 1};

//     auto state = initial_state;
//     Trajectory<StateType> trajectory;

//     double current_time_s{0.0};

//     for (size_t i = 0; i < horizon_; ++i)
//     {
//       ActionType action;
//       action.steering_rate = d(gen);
//       action.jerk = d(gen);

//       state = DynamicsT::step(state, action);
//       trajectory.states.push_back(state);
//       trajectory.time.push_back(current_time_s);
//       current_time_s += DynamicsT::ts;
//     }
//     return trajectory;
//   };

// private:
//   int num_iters_;
//   int horizon_;
//   int population_;
// };

} // namespace Magnum::Examples