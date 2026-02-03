import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from flatland.callbacks.generate_movie_callbacks import GenerateMovieCallbacks
from flatland.env_generation.env_generator import env_generator
from flatland.envs.observations import FullEnvObservation
from flatland.envs.rewards import DefaultRewards
from flatland.trajectories.policy_runner import PolicyRunner
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy


@pytest.mark.parametrize("scale_max_episode_steps,expected", [(1, 4 / 7), (2, 1.0)])
def test_intermediate(scale_max_episode_steps, expected, gen_movies=False, debug=False):
    rewards = DefaultRewards(intermediate_not_served_penalty=0.77,
                             cancellation_factor=22,
                             intermediate_late_arrival_penalty_factor=33,
                             intermediate_early_departure_penalty_factor=44,
                             )
    env, _, _ = env_generator(
        n_cities=5,
        line_length=3,
        obs_builder_object=FullEnvObservation(),
        seed=982374,
        rewards=rewards
    )
    for a in env.agents:
        print(f"agent {a.handle}:")
        print(f" {a.waypoints}")
        print(f" {a.waypoints_earliest_departure}")
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_data_dir = Path(tmpdirname)
        env._max_episode_steps = env._max_episode_steps * scale_max_episode_steps
        trajectory = PolicyRunner.create_from_policy(
            policy=DeadLockAvoidancePolicy(use_alternative_at_first_intermediate_and_then_always_first_strategy=3),
            data_dir=temp_data_dir,
            env=env,
            snapshot_interval=0,
            ep_id=str(uuid.uuid4()),
            callbacks=GenerateMovieCallbacks() if gen_movies else None,
        )
        assert np.isclose(trajectory.trains_arrived["success_rate"], expected)
        if debug:
            for agent_id, a in enumerate(env.agents):
                print(a.waypoints)
                for env_time in range(1, env._elapsed_steps + 1):
                    print(trajectory.position_lookup(env_time, agent_id))
