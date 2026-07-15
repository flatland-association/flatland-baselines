import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.grid.rail_env_grid import RailEnvTransitions, RailEnvTransitionsEnum
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState
from flatland.envs.timetable_generators import ttgen_flatland2
from flatland.envs.timetable_utils import Line
from flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation import FullEnvObservation
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy

# A simple switch (T-junction) merging a north arm and an east arm onto a single west arm:
#
#   column:    0                     1                          2
#   row 0:                      dead-end (north arm)
#   row 1:     dead-end (west arm)   simple_switch_east_left   dead-end (east arm)
#
# A train arriving from the west arm (moving east) can go straight through to the east arm, or
# branch left onto the north arm -- that is the "left" in `simple_switch_east_left`. A train coming
# down the north arm, and a train coming in from the east arm, both merge onto that same west arm.
NORTH_ARM = (0, 1)
WEST_ARM = (1, 0)
SWITCH = (1, 1)
EAST_ARM = (1, 2)


def _make_switch_rail() -> RailGridTransitionMap:
    dead_end_from_south = RailEnvTransitionsEnum.dead_end_from_south
    dead_end_from_west = RailEnvTransitionsEnum.dead_end_from_west
    dead_end_from_east = RailEnvTransitionsEnum.dead_end_from_east
    simple_switch_east_left = RailEnvTransitionsEnum.simple_switch_east_left
    rail_map = np.array([
        [0, dead_end_from_south, 0],
        [dead_end_from_east, simple_switch_east_left, dead_end_from_west],
    ], dtype=np.uint16)
    return RailGridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=RailEnvTransitions(), grid=rail_map)


def _converging_agents_line_generator(_rail, _num_agents, _hints, _num_resets, _np_random) -> Line:
    # agent 0 approaches the switch from the north arm, agent 1 from the east arm; both merge onto
    # the same west arm through the switch cell.
    return Line(
        agent_waypoints={
            0: [[Waypoint(NORTH_ARM, int(Grid4TransitionsEnum.NORTH))], [Waypoint(WEST_ARM, None)]],
            1: [[Waypoint(EAST_ARM, int(Grid4TransitionsEnum.EAST))], [Waypoint(WEST_ARM, None)]],
        },
        agent_speeds=[1.0, 1.0],
    )


def _build_env() -> RailEnv:
    rail = _make_switch_rail()
    env = RailEnv(
        width=rail.width,
        height=rail.height,
        rail_generator=rail_from_grid_transition_map(rail),
        line_generator=_converging_agents_line_generator,
        # both agents have earliest departure 0, so they enter and reach the switch as early as possible.
        timetable_generator=ttgen_flatland2,
        number_of_agents=2,
        obs_builder_object=FullEnvObservation(),
    )
    env.reset()
    return env


def test_two_agents_at_switch_non_facing():
    """
    Two agents each sit one cell ahead of a simple switch non-facing (`simple_switch_east_left`): agent 0 coming
    down the north arm, agent 1 coming in from the east arm; both merge onto the same west arm
    through the switch cell. Once both are in place, DeadLockAvoidancePolicy issues MOVE_FORWARD for
    both -- it only flags an opposing agent if one is already occupying a cell on the agent's own
    path, and since neither agent currently occupies the (still empty) switch cell, no conflict is
    detected between them. It is the environment's own motion check (not the policy) that actually
    prevents two agents from occupying the same cell in the same time step: only one of them ends up
    moving into the switch cell, the other is stopped in place for that step.
    """
    env = _build_env()
    policy = DeadLockAvoidancePolicy(use_entering_prevention=False, min_free_cell=1)
    observations = env._get_observations()

    # run until both agents have entered and are sitting one cell ahead of the switch.
    for _ in range(10):
        if env.agents[0].position == NORTH_ARM and env.agents[1].position == EAST_ARM:
            break
        action_dict = policy.act_many(env.get_agent_handles(), observations=list(observations.values()))
        observations, _, _, _ = env.step(action_dict)
    else:
        raise AssertionError("expected both agents to reach their starting positions ahead of the switch within 10 steps")

    # the tick where both agents attempt to move into the switch cell at the same time.
    action_dict = policy.act_many(env.get_agent_handles(), observations=list(observations.values()))
    assert action_dict[0] == RailEnvActions.MOVE_FORWARD
    assert action_dict[1] == RailEnvActions.MOVE_FORWARD

    env.step(action_dict)

    positions = [agent.position for agent in env.agents]
    states = [agent.state for agent in env.agents]

    assert positions.count(SWITCH) == 1, "expected exactly one agent to have moved into the switch cell"
    moved = 0 if positions[0] == SWITCH else 1
    blocked = 1 - moved

    expected_blocked_position = NORTH_ARM if blocked == 0 else EAST_ARM
    assert positions[blocked] == expected_blocked_position, "expected the blocked agent to stay in place"
    assert states[blocked] == TrainState.STOPPED, "expected the blocked agent to be stopped by the environment's motion check"
