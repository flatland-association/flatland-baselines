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

# Two switches A and B connected by a single-track trunk, each with its own north arm and an outward
# arm on the far side:
#
#                N_OF_A                                            N_OF_B
#                  |                                                  |
#                  v                                                  v
#   WEST_OF_A -- [A: simple_switch_west_right] -- trunk (3 cells) -- [B: simple_switch_east_left] -- EAST_OF_B
#
# Agent 0 starts at N_OF_A and targets EAST_OF_B: N_OF_A -> A -> trunk -> B -> EAST_OF_B.
# Agent 1 starts at N_OF_B and targets WEST_OF_A: N_OF_B -> B -> trunk -> A -> WEST_OF_A.
# Both agents' paths traverse the entire trunk, in opposite directions.
N_OF_A = (0, 1)
N_OF_B = (0, 5)
WEST_OF_A = (1, 0)
SWITCH_A = (1, 1)
TRUNK = [(1, 2), (1, 3), (1, 4)]
SWITCH_B = (1, 5)
EAST_OF_B = (1, 6)


def _make_two_switches_rail() -> RailGridTransitionMap:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_west_right = int(RailEnvTransitionsEnum.simple_switch_west_right)
    simple_switch_east_left = int(RailEnvTransitionsEnum.simple_switch_east_left)

    row0 = [0, dead_end_from_south, 0, 0, 0, dead_end_from_south, 0]
    row1 = [dead_end_from_east, simple_switch_west_right] + [horizontal_straight] * len(TRUNK) + [simple_switch_east_left, dead_end_from_west]
    rail_map = np.array([row0, row1], dtype=np.uint16)
    rail = RailGridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    return rail


def _facing_agents_line_generator(rail, num_agents, hints, num_resets, np_random):
    # agent 0 enters via switch A from the north and targets beyond switch B (east);
    # agent 1 enters via switch B from the north and targets beyond switch A (west).
    return Line(
        agent_waypoints={
            0: [[Waypoint(N_OF_A, int(Grid4TransitionsEnum.NORTH))], [Waypoint(EAST_OF_B, None)]],
            1: [[Waypoint(N_OF_B, int(Grid4TransitionsEnum.NORTH))], [Waypoint(WEST_OF_A, None)]],
        },
        agent_speeds=[1.0, 1.0],
    )


def _build_env() -> RailEnv:
    rail = _make_two_switches_rail()
    env = RailEnv(
        width=rail.width,
        height=rail.height,
        rail_generator=rail_from_grid_transition_map(rail),
        line_generator=_facing_agents_line_generator,
        # both agents have earliest departure 0, so they enter and reach their switch as early as possible.
        timetable_generator=ttgen_flatland2,
        number_of_agents=2,
        obs_builder_object=FullEnvObservation(),
    )
    env.reset()
    return env


def test_two_agents_enter_facing_switches_and_deadlock_there():
    """
    Two agents approach the single-track trunk between switch A and switch B from opposite ends,
    each via its own north arm, and each targeting the cell beyond the *other* switch. Both agents
    reach their own switch cell, but neither ever advances onto the shared trunk: by the time each
    considers stepping onto the trunk, the other is already sitting on its switch cell -- which is a
    cell on its own path -- facing the opposite direction, so DeadLockAvoidancePolicy correctly
    detects the conflict and stops both agents right there. Since their paths are exact mirror images
    of one another over the entire trunk, no number of free cells or trunk length changes this: the
    two agents permanently deadlock at their own switch cells, each blocked by the other being on its
    path, without ever entering the trunk between them.
    """
    env = _build_env()
    policy = DeadLockAvoidancePolicy(use_entering_prevention=False, min_free_cell=1)
    observations = env._get_observations()

    reached_switch = {0: False, 1: False}
    entered_trunk = {0: False, 1: False}
    for _ in range(20):
        action_dict = policy.act_many(env.get_agent_handles(), observations=list(observations.values()))
        observations, _, dones, _ = env.step(action_dict)
        for handle, expected_switch in ((0, SWITCH_A), (1, SWITCH_B)):
            position = env.agents[handle].position
            if position == expected_switch:
                reached_switch[handle] = True
            if position in TRUNK:
                entered_trunk[handle] = True

    assert reached_switch[0] and reached_switch[1], "expected both agents to reach their own switch"
    assert not entered_trunk[0] and not entered_trunk[1], "expected neither agent to ever enter the trunk"

    positions = [agent.position for agent in env.agents]
    states = [agent.state for agent in env.agents]
    assert positions == [SWITCH_A, SWITCH_B], "expected both agents to be stuck at their own switch cell"
    assert states == [TrainState.STOPPED, TrainState.STOPPED], "expected both agents to be permanently stopped"
    assert not dones["__all__"], "expected the agents to deadlock and never both arrive"


# Same infrastructure as above, but with two more agents queuing up north of switch B, behind the
# first one, all sharing the same route/target as the first (beyond switch A, to the west):
#
#                                                        Q_CAP        (dead end, unused)
#                                                          |
#                                                        Q_THIRD      (agent 3 starts here)
#                                                          |
#                                                        Q_SECOND     (agent 2 starts here)
#                                                          |
#              Q_N_OF_A                                 Q_FIRST       (agent 1 starts here)
#                 |                                        |
#                 v                                        v
#  Q_WEST_OF_A -- [A: simple_switch_west_right] -- trunk -- [B: simple_switch_east_left] -- Q_EAST_OF_B
Q_N_OF_A = (3, 1)
Q_WEST_OF_A = (4, 0)
Q_SWITCH_A = (4, 1)
Q_TRUNK = [(4, 2), (4, 3), (4, 4)]
Q_SWITCH_B = (4, 5)
Q_EAST_OF_B = (4, 6)
Q_FIRST = (3, 5)
Q_SECOND = (2, 5)
Q_THIRD = (1, 5)


def _make_two_switches_with_queue_rail() -> RailGridTransitionMap:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_west_right = int(RailEnvTransitionsEnum.simple_switch_west_right)
    simple_switch_east_left = int(RailEnvTransitionsEnum.simple_switch_east_left)

    row0 = [0, 0, 0, 0, 0, dead_end_from_south, 0]
    row1 = [0, 0, 0, 0, 0, vertical_straight, 0]
    row2 = [0, 0, 0, 0, 0, vertical_straight, 0]
    row3 = [0, dead_end_from_south, 0, 0, 0, vertical_straight, 0]
    row4 = [dead_end_from_east, simple_switch_west_right] + [horizontal_straight] * len(Q_TRUNK) + [simple_switch_east_left, dead_end_from_west]
    rail_map = np.array([row0, row1, row2, row3, row4], dtype=np.uint16)
    rail = RailGridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    return rail


def _facing_agents_with_queue_line_generator(rail, num_agents, hints, num_resets, np_random):
    # agent 0 enters via switch A from the north and targets beyond switch B (east), same as before.
    # agents 1, 2 and 3 queue up north of switch B, all sharing agent 1's route: via switch B, the
    # trunk and switch A, to beyond switch A (west).
    return Line(
        agent_waypoints={
            0: [[Waypoint(Q_N_OF_A, int(Grid4TransitionsEnum.NORTH))], [Waypoint(Q_EAST_OF_B, None)]],
            1: [[Waypoint(Q_FIRST, int(Grid4TransitionsEnum.SOUTH))], [Waypoint(Q_WEST_OF_A, None)]],
            2: [[Waypoint(Q_SECOND, int(Grid4TransitionsEnum.SOUTH))], [Waypoint(Q_WEST_OF_A, None)]],
            3: [[Waypoint(Q_THIRD, int(Grid4TransitionsEnum.SOUTH))], [Waypoint(Q_WEST_OF_A, None)]],
        },
        agent_speeds=[1.0, 1.0, 1.0, 1.0],
    )


def _build_env_with_queue() -> RailEnv:
    rail = _make_two_switches_with_queue_rail()
    env = RailEnv(
        width=rail.width,
        height=rail.height,
        rail_generator=rail_from_grid_transition_map(rail),
        line_generator=_facing_agents_with_queue_line_generator,
        # all agents have earliest departure 0, so they enter and reach their switch/queue as early as possible.
        timetable_generator=ttgen_flatland2,
        number_of_agents=4,
        obs_builder_object=FullEnvObservation(),
    )
    env.reset()
    return env


def test_agents_queuing_behind_a_blocked_leader_are_also_stopped_directly():
    """
    Extends the scenario above with two more agents (2 and 3) queuing up behind agent 1, north of
    switch B, all three sharing the exact same route and target (beyond switch A, to the west).

    One might expect that once agent 0 and agent 1 block each other exactly as in the scenario above,
    agents 2 and 3 would keep receiving MOVE_FORWARD from DeadLockAvoidancePolicy -- since they are not
    literally facing an oncoming train yet -- and only be brought to a halt by the environment's own
    motion check as they catch up to the queue, one cell behind agent 1 each. That is not what happens:
    DeadLockAvoidancePolicy's opposition check is evaluated independently per agent along that agent's
    own full path, and agents 2 and 3 share agent 1's route all the way through switch A -- the exact
    cell where agent 0 sits. So agents 2 and 3 each independently detect agent 0 as oncoming on their
    own path too, and DeadLockAvoidancePolicy issues STOP_MOVING for all four agents at once, well
    before agents 2 and 3 have caught up to form a physical queue behind agent 1.
    """
    env = _build_env_with_queue()
    policy = DeadLockAvoidancePolicy(use_entering_prevention=False, min_free_cell=1)
    observations = env._get_observations()

    action_dict = None
    for _ in range(20):
        action_dict = policy.act_many(env.get_agent_handles(), observations=list(observations.values()))
        observations, _, dones, _ = env.step(action_dict)

    assert all(action == RailEnvActions.STOP_MOVING for action in action_dict.values()), \
        "expected DeadLockAvoidancePolicy to stop all four agents directly, not just the two facing each other"

    positions = [agent.position for agent in env.agents]
    states = [agent.state for agent in env.agents]
    assert positions == [Q_SWITCH_A, Q_SWITCH_B, Q_FIRST, Q_SECOND], \
        "expected agents 2 and 3 to be stopped one cell short of where they started, never having caught up to the queue"
    assert states == [TrainState.STOPPED] * 4, "expected all four agents to be permanently stopped"
    assert not dones["__all__"], "expected the agents to deadlock and never all arrive"
