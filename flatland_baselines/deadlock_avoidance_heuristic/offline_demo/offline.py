from pathlib import Path

import numpy as np
import pytest
from flatland.trajectories.policy_grid_runner import generate_trajectories_from_metadata

from flatland.evaluators.trajectory_analysis import data_frame_for_trajectories

if __name__ == '__main__':
    data_dir = Path("./results_environments_v2").resolve()
    if True:
        metadata_csv = Path("/Users/che/workspaces/flatland-scenarios/trajectories/malfunction_deadlock_avoidance_heuristics/metadata.csv").resolve()

        data_dir.mkdir()
        with pytest.raises(SystemExit) as e_info:
            generate_trajectories_from_metadata([
                "--metadata-csv", metadata_csv,
                "--data-dir", data_dir,
                "--policy-pkg", "flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy",
                "--policy-cls", "DeadLockAvoidancePolicy",
                "--obs-builder-pkg", "flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation",
                "--obs-builder-cls", "FullEnvObservation",
                "--callbacks-pkg", "flatland.callbacks.generate_movie_callbacks", "--callbacks-cls", "GenerateMovieCallbacks"
            ])
        assert e_info.value.code == 0
    if True:
        all_actions, all_trains_positions, all_trains_arrived, all_trains_rewards_dones_infos, env_stats, agent_stats = data_frame_for_trajectories(
            root_data_dir=Path(data_dir))
        print(all_trains_arrived)

        sum_normalized_reward = all_trains_arrived["mean_normalized_reward"].sum()
        mean_normalized_reward = all_trains_arrived["mean_normalized_reward"].mean()
        mean_percentage_complete = all_trains_arrived["success_rate"].mean()
        mean_reward = all_trains_rewards_dones_infos.groupby(['episode_id']).agg({"reward": "sum"}).mean()['reward']

        # Round off the reward values, see service.py
        mean_reward = round(mean_reward, 2)
        mean_normalized_reward = round(mean_normalized_reward, 5)
        mean_percentage_complete = round(mean_percentage_complete, 3)

        print(f"# Mean Reward : {mean_reward}")
        print(f"# Sum Normalized Reward : {sum_normalized_reward} (primary score)")
        print(f"# Mean Percentage Complete : {mean_percentage_complete} (secondary score)")
        print(f"# Mean Normalized Reward : {mean_normalized_reward}")

        # TODO rounding problem?
        assert mean_reward == -3541.52
        assert np.isclose(43.08898598301832, sum_normalized_reward)
        assert mean_percentage_complete == 0.678
        assert mean_normalized_reward == 0.86178


