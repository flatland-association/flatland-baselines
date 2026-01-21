import tempfile
import uuid
from pathlib import Path

import numpy as np

from flatland.env_generation.env_generator import env_generator
from flatland.envs.observations import FullEnvObservation
from flatland.envs.rewards import DefaultRewards
from flatland.trajectories.policy_runner import PolicyRunner
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy


def test_intermediate():
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
        print(a.waypoints)
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_data_dir = Path(tmpdirname)
        trajectory = PolicyRunner.create_from_policy(
            policy=DeadLockAvoidancePolicy(),
            data_dir=temp_data_dir,
            env=env,
            snapshot_interval=0,
            ep_id=str(uuid.uuid4())
        )
        assert np.isclose(trajectory.trains_arrived["success_rate"], 1.0)

        # TODO review design decision: vanish immediately at target?
        assert min(env.agents[6].latest_arrival - env.agents[6].arrival_time, 0) == -2
        assert trajectory.trains_rewards_dones_infos["reward"].sum() == - rewards.intermediate_not_served_penalty * env.number_of_agents - 2

        for agent_id, a in enumerate(env.agents):
            print(a.waypoints)
            for env_time in range(1, env._elapsed_steps + 1):
                print(trajectory.position_lookup(env_time, agent_id))
