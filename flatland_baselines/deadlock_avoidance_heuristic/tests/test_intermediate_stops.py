import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from flatland.callbacks.generate_movie_callbacks import GenerateMovieCallbacks
from flatland.env_generation.env_generator import env_generator_legacy
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
    env, _, _ = env_generator_legacy(
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


def test_intermediate_service_audit_trail_records_blocked_conflict():
    """
    Confirms the DLA collision-avoidance decisions themselves are correct after the `StartStepService`
    extraction: "agent X blocked by Y" entries are appended inside `StartStepService._check_agent_can_move`,
    so they land in `policy.start_step_service.audit`, not `policy.audit` -- `DeadLockAvoidancePolicy` and
    `StartStepService` intentionally keep independent audit lists (see
    `test_start_step_service.test_policy_and_service_keep_independent_audit_lists`).

    With the exact same deterministic scenario and seed used here, agent 2 is genuinely oncoming to agent
    4 starting at env_time 14 and must be recorded as blocked.
    """
    rewards = DefaultRewards(intermediate_not_served_penalty=0.77,
                             cancellation_factor=22,
                             intermediate_late_arrival_penalty_factor=33,
                             intermediate_early_departure_penalty_factor=44,
                             )
    env, _, _ = env_generator_legacy(
        n_cities=5,
        line_length=3,
        obs_builder_object=FullEnvObservation(),
        seed=982374,
        rewards=rewards
    )
    policy = DeadLockAvoidancePolicy(use_alternative_at_first_intermediate_and_then_always_first_strategy=3, audit=True, seed=42)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # policy.start_step_service.audit is a plain list, independent of `trajectory.trains_arrived`'s pandas dtypes.
        PolicyRunner.create_from_policy(
            policy=policy,
            data_dir=Path(tmpdirname),
            env=env,
            snapshot_interval=0,
            ep_id=str(uuid.uuid4()),
        )
        assert any(
            entry["env_time"] == 14 and entry["agent_id"] == 2 and "blocked by 4" in entry["v"]
            for entry in policy.start_step_service.audit
        ), "expected agent 2 to be recorded as blocked by agent 4 at env_time 14"
