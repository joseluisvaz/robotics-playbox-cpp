





template <typename DynamicsT>
class CEM_MPC
{

public:
  CEM_MPC() = default;
  CEM_MPC(const int num_iters, const int horizon, const int population, const int elites)
      : num_iters_(num_iters), horizon_(horizon), population_(population), elites_(elites)
  {
    trajectory_.states = EigenStateSequence::Zero(DynamicsT::state_size, horizon_);
    trajectory_.actions = EigenActionSequence::Zero(DynamicsT::action_size, horizon_);
    trajectory_.times = std::vector<float>(horizon_, 0.0f);

    costs_index_pair_ = std::vector<std::pair<float, int>>(population_, std::make_pair(0.0f, 0));

    for (int i = 0; i < population_; ++i)
    {
      EigenTrajectory trajectory;
      trajectory.states = EigenStateSequence::Zero(DynamicsT::state_size, horizon_);
      trajectory.actions = EigenActionSequence::Zero(DynamicsT::action_size, horizon_);
      trajectory.times = std::vector<float>(horizon_, 0.0f);
      candidate_trajectories_.emplace_back(std::move(trajectory));
    }

    EigenActionSequence mean = EigenActionSequence::Zero(trajectory_.actions.rows(), trajectory_.actions.cols());
    EigenActionSequence stddev = EigenActionSequence::Ones(trajectory_.actions.rows(), trajectory_.actions.cols());

    sampler_ = NormalRandomVariable(mean, stddev);
  };

  void rollout(EigenTrajectory &trajectory)
  {
    double current_time_s{0.0};
    for (size_t i = 0; i + 1 < horizon_; ++i)
    {
      trajectory.times.at(i) = current_time_s;

      DynamicsT::step(trajectory.states.col(i), trajectory.actions.col(i), trajectory.states.col(i + 1));
      current_time_s += DynamicsT::ts;
    }
    trajectory.times.at(horizon_ - 1) = current_time_s;
  }

  void run_cem_iteration(const Ref<EigenState> &initial_state)
  {
    EASY_FUNCTION(profiler::colors::Yellow);

    const auto evaluate_trajectory_fn = [this, &initial_state](int k)
    {
      int block_size = static_cast<int>(this->population_ / this->threads_.size());
      for (int j = block_size * k; j < block_size * k + block_size; ++j)
      {
        auto &trajectory = this->candidate_trajectories_.at(j);
        trajectory.states.col(0) = initial_state; // Set first state
        trajectory.actions = this->sampler_();    // Sample actions
        this->rollout(trajectory);

        this->costs_index_pair_.at(j).first = this->cost_function_(trajectory.states, trajectory.actions);
        this->costs_index_pair_.at(j).second = j;
      }
    };

    for (int k = 0; k < threads_.size(); k++)
    {
      threads_.at(k) = std::thread([&evaluate_trajectory_fn, k]() { evaluate_trajectory_fn(k); });
    }

    for (auto &&thread : threads_)
    {
      if (thread.joinable())
      {
        thread.join();
      }
    }

    std::sort(costs_index_pair_.begin(), costs_index_pair_.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    EigenActionSequence mean_actions = EigenActionSequence::Zero(action_size, horizon_);
    for (int i = 0; i < elites_; ++i)
    {
      const auto &elite_index = costs_index_pair_.at(i).second;
      mean_actions += candidate_trajectories_.at(elite_index).actions;
    }

    mean_actions = mean_actions / elites_;

    EigenActionSequence stddev = EigenActionSequence::Zero(action_size, horizon_);
    for (int i = 0; i < elites_; ++i)
    {
      const auto &elite_index = costs_index_pair_.at(i).second;
      EigenActionSequence temp = candidate_trajectories_.at(elite_index).actions - mean_actions;
      stddev = stddev + temp.cwiseProduct(temp);
    }

    stddev = (stddev / elites_).cwiseSqrt();

    sampler_.mean = mean_actions;
    sampler_.transform = stddev;
  }

  EigenTrajectory &execute(const Ref<EigenState> &initial_state)
  {
    EASY_FUNCTION(profiler::colors::Magenta);

    sampler_.mean = EigenActionSequence::Zero(trajectory_.actions.rows(), trajectory_.actions.cols());
    sampler_.transform = EigenActionSequence::Ones(trajectory_.actions.rows(), trajectory_.actions.cols());

    for (int i = 0; i < num_iters_; ++i)
    {
      run_cem_iteration(initial_state);
    }

    trajectory_.states.col(0) = initial_state; // Set first state

    trajectory_.actions = sampler_.mean; // Set mean actions
    rollout(trajectory_);
    return trajectory_;
  };

  CostFunction cost_function_;

private:
  int num_iters_;
  int horizon_;
  int population_;
  int elites_;
  std::vector<EigenTrajectory> candidate_trajectories_;
  EigenTrajectory trajectory_;
  EigenTrajectory final_trajectory;
  NormalRandomVariable sampler_;
  std::vector<std::thread> threads_{16};
  std::vector<std::pair<float, int>> costs_index_pair_;
};