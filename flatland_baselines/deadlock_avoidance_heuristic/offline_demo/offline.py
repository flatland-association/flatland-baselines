from pathlib import Path

from flatland.evaluators.trajectory_analysis import data_frame_for_trajectories

if __name__ == '__main__':
    data_dir = Path("./results_environments_v2").resolve()
    if False:
        metadata_csv = Path("/Users/che/workspaces/flatland-scenarios/scenarios/metadata.csv").resolve()

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
        print(all_trains_arrived["mean_normalized_reward"].sum())
        print(all_trains_arrived["mean_normalized_reward"].mean())
        print(all_trains_arrived["success_rate"].mean())
        # TODO take mean of summs grouped by test/level
        print(all_trains_rewards_dones_infos["reward"].sum())
