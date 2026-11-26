import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from flatland.env_generation.env_generator import env_generator_legacy
from flatland.envs.persistence import RailEnvPersister
from flatland.trajectories.policy_runner import generate_trajectory_from_policy


@pytest.mark.parametrize(
    "seed, expected",
    [
        # NOTE: seed=1002's reward/normalized_reward re-calibrated for
        # flatland-rl@178-agents-living-on-the-edge-14's earliest_departure=0 dispatch-timing fix
        # (issue #280): one step earlier, -7 -> -6. seed=1003 and None unaffected.
        (1002, {'normalized_reward': -0.0030287733467945007 + 1, 'percentage_complete': 1.0, 'reward': -6, 'termination_cause': None, }),
        (1003, {'normalized_reward': -0.0095911155981827365 + 1, 'percentage_complete': 1.0, 'reward': -19, 'termination_cause': None}),
        (None, {'normalized_reward': 0.0 + 1, 'termination_cause': None, 'reward': 0, 'percentage_complete': 1.0}),
    ])
def test_env_path_and_seed(seed, expected):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        env_file = str(tmp_dir / "env.pkl")
        data_dir = tmp_dir / "data_dir"
        data_dir.mkdir()
        # TODO https://github.com/flatland-association/flatland-rl/issues/242 rail_generator etc. not persisted, the outcome should not depend on this seed, but it currently does. In particular, seed same seed here and in --seed (pased to reset()) should have the same outcome.
        env, _, _ = env_generator_legacy(seed=1001)
        scenario_id = uuid.uuid4()
        RailEnvPersister.save(env, env_file)

        with pytest.raises(SystemExit) as e_info:
            args = [
                "--data-dir", data_dir,
                "--ep-id", scenario_id,
                "--env-path", env_file,
                "--policy", "flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy.DeadLockAvoidancePolicy",
                "--obs-builder", "flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation.FullEnvObservation",
                "--callbacks", "flatland.evaluators.evaluator_callback.FlatlandEvaluatorCallbacks",
                "--snapshot-interval", "-1",
            ]
            if seed is not None:
                args += ["--post-seed", str(seed)]
            generate_trajectory_from_policy(args)
        assert e_info.value.code == 0
        with (data_dir / "outputs" / "evaluation.json").open("r") as f:
            actual = json.load(f)
        print(actual)
        assert actual['termination_cause'] == expected['termination_cause'], ('termination_cause', actual, expected)
        for c in ['normalized_reward', 'percentage_complete', 'reward']:
            assert np.isclose(actual[c], expected[c]), (c, actual, expected)
