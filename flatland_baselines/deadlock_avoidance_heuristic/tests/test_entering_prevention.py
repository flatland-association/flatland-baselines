from typing import Dict, Optional, Tuple

import numpy as np
import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.grid.rail_env_grid import RailEnvTransitions, RailEnvTransitionsEnum
from flatland.envs.line_generators import LineGenerator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.timetable_generators import ttgen_flatland2
from flatland.envs.timetable_utils import Line
from flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation import FullEnvObservation
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy

# A single row of track, 7 cells wide, with an unused dead end at each far end (column 0 and column
# 6) so that A and B themselves are plain straight cells (not dead ends), each with a straight
# segment of track on either side:
#
#   column:    0         1   2   3   4   5         6
#              dead-end  A   .   .   .   B   dead-end
#
# This matters: at a dead-end cell only one direction is ever valid, so an agent resting at its own
# home dead-end is forced to have the same direction value as another agent's path arriving there,
# which can hide a head-on conflict from the opposition check. A and B being ordinary straight cells
# lets each agent be given its true direction of travel, so opposing agents are always detected correctly.
TRACK_LENGTH = 7
A = (0, 1)
B = (0, TRACK_LENGTH - 2)

# The same scenario, but with A and B being the dead ends themselves, at the very ends of the track:
# dead-end A --- ... --- B dead-end. See `test_with_entering_prevention_and_dead_ends_still_deadlocks`
# below for why this variant is a known limitation.
DEAD_END_TRACK_LENGTH = 5
DEAD_END_A = (0, 0)
DEAD_END_B = (0, DEAD_END_TRACK_LENGTH - 1)


def _make_single_track_rail(length: int) -> RailGridTransitionMap:
    dead_end_from_west = RailEnvTransitionsEnum.dead_end_from_west
    dead_end_from_east = RailEnvTransitionsEnum.dead_end_from_east
    horizontal_straight = RailEnvTransitionsEnum.horizontal_straight
    rail_map = np.array([[dead_end_from_east] + [horizontal_straight] * (length - 2) + [dead_end_from_west]], dtype=np.uint16)
    return RailGridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=RailEnvTransitions(), grid=rail_map)


def _opposing_agents_line_generator(a: Tuple[int, int], b: Tuple[int, int], direction_a: int, direction_b: int) -> LineGenerator:
    # agent 0 travels a -> b, agent 1 travels b -> a on the same single track.
    def generate(_rail, _num_agents, _hints, _num_resets, _np_random):
        return Line(
            agent_waypoints={
                0: [[Waypoint(a, int(direction_a))], [Waypoint(b, None)]],
                1: [[Waypoint(b, int(direction_b))], [Waypoint(a, None)]],
            },
            agent_speeds=[1.0, 1.0],
        )

    return generate


def _build_env(length: int, a: Tuple[int, int], b: Tuple[int, int], direction_a: int, direction_b: int) -> RailEnv:
    rail = _make_single_track_rail(length)
    env = RailEnv(
        width=rail.width,
        height=rail.height,
        rail_generator=rail_from_grid_transition_map(rail),
        line_generator=_opposing_agents_line_generator(a, b, direction_a, direction_b),
        # both agents have earliest departure 0, so they are both ready to depart from the first step.
        timetable_generator=ttgen_flatland2,
        number_of_agents=2,
        obs_builder_object=FullEnvObservation(),
    )
    env.reset()
    return env


def _run(env: RailEnv, policy: DeadLockAvoidancePolicy, max_steps: int) -> Tuple[Dict[int, Optional[int]], Dict[int, Optional[int]], bool]:
    """Runs the policy on the env and records, per agent, the step it entered the map and the step it left it (arrived)."""
    observations = env._get_observations()
    entered_at = {0: None, 1: None}
    left_at = {0: None, 1: None}
    all_done = False
    for step in range(max_steps):
        action_dict = policy.act_many(env.get_agent_handles(), observations=list(observations.values()))
        observations, _, dones, _ = env.step(action_dict)
        for handle, agent in enumerate(env.agents):
            if entered_at[handle] is None and agent.position is not None:
                entered_at[handle] = step
            if left_at[handle] is None and entered_at[handle] is not None and dones[handle]:
                left_at[handle] = step
        all_done = dones["__all__"]
        if all_done:
            break
    return entered_at, left_at, all_done


def test_without_entering_prevention_simultaneous_entry_deadlocks():
    """
    Two agents travelling the same single track in opposite directions (A->B and B->A) both become
    ready to depart at the same time step. Without `use_entering_prevention`, both enter at the same
    time step and meet head-on: since there is no passing loop, neither can move again -- a permanent
    deadlock and neither agent ever arrives.
    """
    env = _build_env(TRACK_LENGTH, A, B, Grid4TransitionsEnum.EAST, Grid4TransitionsEnum.WEST)
    policy = DeadLockAvoidancePolicy(use_entering_prevention=False, min_free_cell=1)

    entered_at, _, all_done = _run(env, policy, max_steps=6 * TRACK_LENGTH)

    assert entered_at[0] is not None and entered_at[1] is not None, "expected both agents to enter"
    assert entered_at[0] == entered_at[1], "expected both agents to enter at the same time step"
    assert not all_done, "expected the agents to deadlock and never both arrive"


def test_with_entering_prevention_only_one_agent_enters_until_the_other_has_left():
    """
    With `use_entering_prevention`, only one of the two agents is allowed to enter when both want to
    enter at the same time step. The other has to wait until the first one has left the scene (i.e.
    arrived at its target) before it is allowed to enter itself.
    """
    env = _build_env(TRACK_LENGTH, A, B, Grid4TransitionsEnum.EAST, Grid4TransitionsEnum.WEST)
    policy = DeadLockAvoidancePolicy(use_entering_prevention=True, min_free_cell=1)

    entered_at, left_at, all_done = _run(env, policy, max_steps=6 * TRACK_LENGTH)

    assert entered_at[0] is not None and entered_at[1] is not None, "expected both agents to eventually enter"
    assert entered_at[0] != entered_at[1], "expected only one agent to enter at the first opportunity"

    first, second = (0, 1) if entered_at[0] < entered_at[1] else (1, 0)
    assert left_at[first] is not None, "expected the first agent to leave the scene (arrive)"
    assert entered_at[second] >= left_at[first], "expected the second agent to enter only once the first has left the scene"

    assert all_done, "expected both agents to arrive"


@pytest.mark.xfail(strict=True, reason="""
    Known limitation: use_entering_prevention only serializes entry correctly when A and B are dead-ends.
    In this case, the held-back agent still enters one
    tick after the other and the two permanently deadlock. See the docstring below for the exact
    mechanism, spelled out at t=0 and t=1.
""")
def test_with_entering_prevention_and_dead_ends_still_deadlocks():
    """
    Same scenario as `test_with_entering_prevention_only_one_agent_enters_until_the_other_has_left`
    (agent 0 travels A->B, agent 1 travels B->A on a single track, both ready to depart at the same
    time step), except A and B are now the dead-end cells at the very ends of the track itself:

        dead-end A --- ... --- B dead-end

    - t=0->1: both agents are `READY_TO_DEPART` simultaneously. The entering-prevention pairwise check
      compares the two candidates via `_get_free`, finds no free cells between their fully-overlapping
      paths, and blocks agent 0. Agent 1 enters, landing on its own home dead-end cell B.
    - t=1->2: agent 0 tries again. Agent 1 is sitting exactly at its own entry cell B, not yet having
      taken its first step onward. The regular opposition check only flags an agent as opposing if its
      direction differs from the direction recorded on my own path at that same cell -- but at a
      dead-end cell there is only one valid transition direction, so agent 1's direction while resting
      at B is forced to equal the direction agent 0's path uses when arriving at B. Same direction =>
      not flagged as opposing => zero opposing agents detected => agent 0's move is approved.

    So agent 0 slips onto the track one tick after agent 1, while agent 1 is still active. From there
    both are on the same loop-free track moving toward each other: agent 0 ends up parked exactly on
    agent 1's target cell A, and the two permanently deadlock.
    """
    env = _build_env(DEAD_END_TRACK_LENGTH, DEAD_END_A, DEAD_END_B, Grid4TransitionsEnum.WEST, Grid4TransitionsEnum.EAST)
    policy = DeadLockAvoidancePolicy(use_entering_prevention=True, min_free_cell=1)

    entered_at, left_at, all_done = _run(env, policy, max_steps=6 * DEAD_END_TRACK_LENGTH)

    assert entered_at[0] is not None and entered_at[1] is not None, "expected both agents to eventually enter"
    assert entered_at[0] != entered_at[1], "expected only one agent to enter at the first opportunity"

    first, second = (0, 1) if entered_at[0] < entered_at[1] else (1, 0)
    assert left_at[first] is not None, "expected the first agent to leave the scene (arrive)"
    assert entered_at[second] >= left_at[first], "expected the second agent to enter only once the first has left the scene"

    assert all_done, "expected both agents to arrive"
