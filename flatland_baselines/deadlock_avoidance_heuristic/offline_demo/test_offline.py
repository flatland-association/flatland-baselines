import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from flatland.evaluators.trajectory_analysis import data_frame_for_trajectories
from flatland.trajectories.policy_grid_runner import generate_trajectories_from_metadata


def test_offline_calibrated_against_online_post_seed_old_envs():
    """
    Verify offline evaluation yields the same result as online evaluation in legacy way with current code base,
    and doing a reset on the loaded env (either using `--post-seed` or `FLATLAND_EVALUATION_RANDOM_SEED`).
    """
    _dir = os.getenv("BENCHMARK_EPISODES_FOLDER")
    assert _dir is not None, _dir
    metadata_csv = (Path(_dir) / "malfunction_deadlock_avoidance_heuristics" / "metadata.csv").resolve()

    with tempfile.TemporaryDirectory() as tmpdirname:
        data_dir = Path(tmpdirname) / "data"
        data_dir.mkdir()
        with pytest.raises(SystemExit) as e_info:
            generate_trajectories_from_metadata([
                "--metadata-csv", metadata_csv,
                "--data-dir", data_dir,
                "--policy-pkg", "flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy",
                "--policy-cls", "DeadLockAvoidancePolicy",
                "--obs-builder-pkg", "flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation",
                "--obs-builder-cls", "FullEnvObservation",
                # environments_v2.zip where generated with old stateful rail generator, therefore, we need to use legacy env generator as we use new envs (and not the pickles from environments_v2.zip) here.
                "--legacy-env-generator", True,
                # --post-seed mimicks online evaluation, which does a reset on the env loaded from pkl using `FLATLAND_EVALUATION_RANDOM_SEED`
                "--post-seed", 1001,
            ])
        assert e_info.value.code == 0
        all_actions, all_trains_positions, all_trains_arrived, all_trains_rewards_dones_infos, env_stats, agent_stats = data_frame_for_trajectories(
            root_data_dir=Path(data_dir))
        print(all_trains_arrived)

        sum_normalized_reward = all_trains_arrived["normalized_reward"].sum()
        mean_normalized_reward = all_trains_arrived["normalized_reward"].mean()
        mean_percentage_complete = all_trains_arrived["success_rate"].mean()
        mean_reward = all_trains_rewards_dones_infos.groupby(['episode_id']).agg({"reward": "sum"}).mean()['reward']
        verify_online_offline_calibration_old_envs(mean_normalized_reward, mean_percentage_complete, mean_reward, sum_normalized_reward)


# old envs means environments_v2.zip, which were generated with the old stateful rail generator.
def verify_online_offline_calibration_old_envs(mean_normalized_reward, mean_percentage_complete, mean_reward,
                                               sum_normalized_reward):
    # Round off the reward values, see service.py
    mean_reward = round(mean_reward, 2)
    mean_normalized_reward = round(mean_normalized_reward, 5)
    mean_percentage_complete = round(mean_percentage_complete, 3)

    print(f"# Mean Reward : {mean_reward}")
    print(f"# Sum Normalized Reward : {sum_normalized_reward} (primary score)")
    print(f"# Mean Percentage Complete : {mean_percentage_complete} (secondary score)")
    print(f"# Mean Normalized Reward : {mean_normalized_reward}")

    # From online evaluation, see online_demo/test_online.py
    # evaluator-1   | # Mean Reward : -3427.52
    # evaluator-1   | # Sum Normalized Reward : 43.141500749639434 (primary score)
    # evaluator-1   | # Mean Percentage Complete : 0.671 (secondary score)
    # evaluator-1   | # Mean Normalized Reward : 0.86283
    assert np.isclose(mean_reward, -3427.52)
    assert np.isclose(sum_normalized_reward, 43.141500749639434)
    assert np.isclose(mean_percentage_complete, 0.671)
    assert np.isclose(mean_normalized_reward, 0.86283)
