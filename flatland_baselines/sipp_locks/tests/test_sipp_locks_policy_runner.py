import inspect
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.trajectories.policy_runner import generate_trajectory_from_policy

# Phase 1: the winner's code runs on flatland-rl 4.2.x only. Gate on the interface it relies on, not on the package
# version (flatland-rl may come through PYTHONPATH, without package metadata): 4.3.0 changed reset() to reset(env).
pytestmark = pytest.mark.skipif(
    len(inspect.signature(ObservationBuilder.reset).parameters) != 1,
    reason="sipp_locks needs flatland-rl 4.2.x (ObservationBuilder.reset() without env)",
)

# Same variable and file as flatland-rl's tox.ini/checks.yml: the public ECML 2026 example scenario
# https://github.com/flatland-association/ecml2026-starterkit/raw/refs/heads/main/reinforcement_learning/sampling/level_0_scenario_1.pkl
ECML2026_PKL = os.environ.get("ECML2026_PKL")

# Recorded on flatland-rl 4.2.6 (Python 3.12): all 6 trains arrive; −200 from the skipped intermediate stops.
EXPECTED = {"normalized_reward": 0.9325236167341431, "percentage_complete": 1.0, "reward": -200.0, "termination_cause": None}


@pytest.mark.skipif(ECML2026_PKL is None, reason="set ECML2026_PKL to the path of level_0_scenario_1.pkl")
def test_level_0_scenario_1():
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        data_dir = Path(tmp_dir_name) / "data_dir"
        data_dir.mkdir()
        with pytest.raises(SystemExit) as e_info:
            generate_trajectory_from_policy([
                "--data-dir", data_dir,
                "--ep-id", "level_0_scenario_1",
                "--env-path", ECML2026_PKL,
                "--policy", "flatland_baselines.sipp_locks.or_policy.PlannerPolicy",
                "--obs-builder", "flatland_baselines.sipp_locks.or_policy.PlannerObsBuilder",
                "--rewards", "flatland.envs.rewards.ECML2026Rewards",
                "--callbacks", "flatland.evaluators.evaluator_callback.FlatlandEvaluatorCallbacks",
                "--snapshot-interval", "-1",
            ])
        assert e_info.value.code == 0
        with (data_dir / "outputs" / "evaluation.json").open("r") as f:
            actual = json.load(f)
        print(actual)
        assert actual["termination_cause"] == EXPECTED["termination_cause"], (actual, EXPECTED)
        for c in ["normalized_reward", "percentage_complete", "reward"]:
            assert np.isclose(actual[c], EXPECTED[c]), (c, actual, EXPECTED)
