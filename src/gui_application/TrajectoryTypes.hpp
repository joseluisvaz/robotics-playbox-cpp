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

#include <array>
#include <random>

namespace Magnum::Examples
{

struct KinematicBicycleState
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  double speed{0.0};
  double accel{0.0};
  double steering{0.0};
};

struct KinematicBicycleAction
{
  double jerk{0.0};
  double steering_rate{0.0};
};

template <typename StateT>
struct Trajectory
{
  using StateType = StateT;

  std::vector<double> time{};
  std::vector<StateType> states{};
};

template <typename StateT, typename ActionT>
struct StateActionTrajectory
{
  using StateType = StateT;
  using ActionType = ActionT;

  std::vector<double> time{};
  std::vector<StateType> states{};
  std::vector<ActionType> actions{};
};

using KinematicBicycleTrajectory = Trajectory<KinematicBicycleState>;

struct KinematicBicycle
{
  constexpr static double ts{0.1};
  constexpr static double one_over_wheelbase{1.0 / 3.0};
  constexpr static double steering_max{0.52};

  using StateType = KinematicBicycleState;
  using ActionType = KinematicBicycleAction;

  static StateType step(const StateType &state, const ActionType &action)
  {
    StateType new_state;

    new_state.x = state.x + ts * state.speed * std::cos(state.yaw);
    new_state.y = state.y + ts * state.speed * std::sin(state.yaw);
    new_state.yaw =
        state.yaw +
        ts * state.speed * one_over_wheelbase *
            std::tan(std::clamp(state.steering, -steering_max, steering_max));
    new_state.speed = state.speed + ts * state.accel;
    new_state.accel = state.accel + ts * action.jerk;
    new_state.steering = state.steering + ts * action.steering_rate;
    return new_state;
  }
};

template <typename DynamicsT>
class CEM_MPC
{
  using StateType = typename DynamicsT::StateType;
  using ActionType = typename DynamicsT::ActionType;

public:
  CEM_MPC() = default;
  CEM_MPC(const int num_iters, const int horizon, const int population)
      : num_iters_(num_iters), horizon_(horizon), population_(population){};

  Trajectory<StateType> rollout(StateType initial_state)
  {
    std::random_device rd{};
    std::mt19937 gen{rd()};

    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the
    // mean
    std::normal_distribution<> d{0, 1};

    auto state = initial_state;
    Trajectory<StateType> trajectory;

    double current_time_s{0.0};

    for (size_t i = 0; i < horizon_; ++i)
    {
      ActionType action;
      action.steering_rate = d(gen);
      action.jerk = d(gen);

      state = DynamicsT::step(state, action);
      trajectory.states.push_back(state);
      trajectory.time.push_back(current_time_s);
      current_time_s += DynamicsT::ts;
    }
    return trajectory;
  };

private:
  int num_iters_;
  int horizon_;
  int population_;
};


} // namespace Magnum::Examples