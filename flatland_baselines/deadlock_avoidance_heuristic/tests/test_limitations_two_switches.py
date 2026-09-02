from typing import Dict, Tuple

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.grid.rail_env_grid import RailEnvTransitionsEnum
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
from flatland_baselines.deadlock_avoidance_heuristic.tests.rail_test_utils import build_rail, target_waypoints


def _run_n_steps(env: RailEnv, policy: DeadLockAvoidancePolicy, num_steps: int) -> Tuple[Dict[int, RailEnvActions], Dict[object, bool]]:
    """Runs the policy on the env for a fixed number of steps and returns the final action_dict and dones."""
    observations = env._get_observations()
    action_dict: Dict[int, RailEnvActions] = {}
    dones: Dict[object, bool] = {}
    for _ in range(num_steps):
        action_dict = policy.act_many(env.get_agent_handles(), observations=list(observations.values()))
        observations, _, dones, _ = env.step(action_dict)
    return action_dict, dones


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
    dead_end_from_south = RailEnvTransitionsEnum.dead_end_from_south
    dead_end_from_west = RailEnvTransitionsEnum.dead_end_from_west
    dead_end_from_east = RailEnvTransitionsEnum.dead_end_from_east
    horizontal_straight = RailEnvTransitionsEnum.horizontal_straight
    simple_switch_west_right = RailEnvTransitionsEnum.simple_switch_west_right
    simple_switch_east_left = RailEnvTransitionsEnum.simple_switch_east_left

    row0 = [0, dead_end_from_south, 0, 0, 0, dead_end_from_south, 0]
    row1 = [dead_end_from_east, simple_switch_west_right] + [horizontal_straight] * len(TRUNK) + [simple_switch_east_left, dead_end_from_west]
    return build_rail([row0, row1])


def _facing_agents_line_generator(rail, _num_agents, _hints, _num_resets, _np_random) -> Line:
    # agent 0 enters via switch A from the north and targets beyond switch B (east);
    # agent 1 enters via switch B from the north and targets beyond switch A (west).
    # speed 0.5, not 1.0: since flatland-rl#178 ("let STOP_MOVING complete an in-flight cell-boundary
    # crossing", 178-agents-living-on-the-edge-9), a full-speed agent's entire one-cell-per-step motion
    # is always "in flight" the instant it starts, so STOP_MOVING can never catch it before completing
    # that cell -- it would advance one cell further than intended every time it's stopped. At half
    # speed, a stopped agent is caught mid-cell (pre_offset + pre_speed < SEGMENT_LENGTH) instead, so it
    # freezes exactly where expected, same as pre-#178. See test_two_agents_enter_facing_switches_and_deadlock_there's
    # docstring for the full reasoning and why widening the rail (rather than slowing the agents) doesn't work.
    return Line(
        agent_waypoints={
            0: [[Waypoint(N_OF_A, int(Grid4TransitionsEnum.NORTH))], target_waypoints(rail, EAST_OF_B)],
            1: [[Waypoint(N_OF_B, int(Grid4TransitionsEnum.NORTH))], target_waypoints(rail, WEST_OF_A)],
        },
        agent_speeds=[0.5, 0.5],
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

    Both agents run at half speed (see `_facing_agents_line_generator`) rather than full speed: since
    flatland-rl#178 ("let STOP_MOVING complete an in-flight cell-boundary crossing",
    178-agents-living-on-the-edge-9), a full-speed agent's one-cell-per-step motion is always already
    "in flight" (pre_offset + pre_speed >= SEGMENT_LENGTH) the instant it starts, so it would complete
    one extra cell -- onto the trunk -- every time DeadLockAvoidancePolicy stops it, no matter how the
    rail is laid out (verified empirically: widening the geometry with extra free cells, or raising
    DeadLockAvoidancePolicy's min_free_cell, doesn't help either, since `_check_agent_can_move` only
    applies min_free_cell once an opposing agent is already registered on the shared path -- it isn't a
    static distance threshold, and both agents here become mutually visible on the same step regardless
    of upstream buffer length). At half speed, a stopped agent is instead caught mid-cell
    (pre_offset + pre_speed < SEGMENT_LENGTH), so it freezes exactly at its switch cell, matching the
    pre-#178 behavior this test documents.
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
            agent = env.agents[handle]
            position = agent.current_entry_point[0] if agent.current_entry_point is not None else None
            if position == expected_switch:
                reached_switch[handle] = True
            if position in TRUNK:
                entered_trunk[handle] = True

    assert reached_switch[0] and reached_switch[1], "expected both agents to reach their own switch"
    assert not entered_trunk[0] and not entered_trunk[1], "expected neither agent to ever enter the trunk"

    positions = [agent.current_entry_point[0] if agent.current_entry_point is not None else None for agent in env.agents]
    states = [agent.state for agent in env.agents]
    assert positions == [SWITCH_A, SWITCH_B], "expected both agents to be stuck at their own switch cell"
    assert states == [TrainState.STOPPED, TrainState.STOPPED], "expected both agents to be permanently stopped"
    assert not dones["__all__"], "expected the agents to deadlock and never both arrive"


def _facing_agents_full_speed_line_generator(rail, _num_agents, _hints, _num_resets, _np_random) -> Line:
    # same waypoints as _facing_agents_line_generator, but at full speed -- see
    # test_stop_moving_advances_full_speed_agent_into_trunk_edge9_regression for why.
    return Line(
        agent_waypoints={
            0: [[Waypoint(N_OF_A, int(Grid4TransitionsEnum.NORTH))], target_waypoints(rail, EAST_OF_B)],
            1: [[Waypoint(N_OF_B, int(Grid4TransitionsEnum.NORTH))], target_waypoints(rail, WEST_OF_A)],
        },
        agent_speeds=[1.0, 1.0],
    )


def _build_full_speed_env() -> RailEnv:
    rail = _make_two_switches_rail()
    env = RailEnv(
        width=rail.width,
        height=rail.height,
        rail_generator=rail_from_grid_transition_map(rail),
        line_generator=_facing_agents_full_speed_line_generator,
        timetable_generator=ttgen_flatland2,
        number_of_agents=2,
        obs_builder_object=FullEnvObservation(),
    )
    env.reset()
    return env


def test_stop_moving_advances_full_speed_agent_into_trunk_edge9_regression():
    """
    Regression test for flatland-rl#178's "let STOP_MOVING complete an in-flight cell-boundary
    crossing" change (178-agents-living-on-the-edge-9). Same topology as
    test_two_agents_enter_facing_switches_and_deadlock_there, but both agents run at full speed
    (1.0) instead of half speed, so an agent is always already "in flight"
    (pre_offset + pre_speed >= SEGMENT_LENGTH) the instant it reaches a cell boundary -- there is no
    earlier step at which DeadLockAvoidancePolicy could brake before reaching it.

    Since flatland-rl#178, a MOVING agent at a cell-exit boundary completes its crossing regardless
    of which action is sent -- STOP_MOVING no longer prevents entry into the next cell, it only fails
    to steer onto the correct branch at a switch (rail_grid_transition_map.py's `_check_action_new`
    silently resolves STOP_MOVING to "continue straight" there, unlike a real MOVE_LEFT/MOVE_RIGHT).
    DeadLockAvoidancePolicy now computes and sends its own intended directional action instead of
    blind STOP_MOVING whenever this is the case (see `_extract_agent_can_move`'s `forced_action`), so
    each agent is steered correctly onto the trunk rather than derailed onto an unplanned branch --
    but, since full speed still forces them one cell further per step before DLA's opposition check
    can react, they end up meeting one cell short of collision inside the trunk (not frozen at their
    own switch cell, as they would at half speed) rather than at their own switch cell.
    """
    env = _build_full_speed_env()
    policy = DeadLockAvoidancePolicy(use_entering_prevention=False, min_free_cell=1)
    observations = env._get_observations()

    for _ in range(20):
        action_dict = policy.act_many(env.get_agent_handles(), observations=list(observations.values()))
        observations, _, dones, _ = env.step(action_dict)

    positions = [agent.current_entry_point[0] if agent.current_entry_point is not None else None for agent in env.agents]
    states = [agent.state for agent in env.agents]
    assert states == [TrainState.STOPPED, TrainState.STOPPED], "expected both agents to be permanently stopped"
    assert positions == [TRUNK[1], TRUNK[2]], \
        "expected both agents correctly routed onto the trunk (not derailed onto the wrong switch " \
        "branch) and deadlocked one cell short of colliding head-on"
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
    dead_end_from_south = RailEnvTransitionsEnum.dead_end_from_south
    dead_end_from_west = RailEnvTransitionsEnum.dead_end_from_west
    dead_end_from_east = RailEnvTransitionsEnum.dead_end_from_east
    vertical_straight = RailEnvTransitionsEnum.vertical_straight
    horizontal_straight = RailEnvTransitionsEnum.horizontal_straight
    simple_switch_west_right = RailEnvTransitionsEnum.simple_switch_west_right
    simple_switch_east_left = RailEnvTransitionsEnum.simple_switch_east_left

    row0 = [0, 0, 0, 0, 0, dead_end_from_south, 0]
    row1 = [0, 0, 0, 0, 0, vertical_straight, 0]
    row2 = [0, 0, 0, 0, 0, vertical_straight, 0]
    row3 = [0, dead_end_from_south, 0, 0, 0, vertical_straight, 0]
    row4 = [dead_end_from_east, simple_switch_west_right] + [horizontal_straight] * len(Q_TRUNK) + [simple_switch_east_left, dead_end_from_west]
    return build_rail([row0, row1, row2, row3, row4])


def _facing_agents_with_queue_line_generator(rail, _num_agents, _hints, _num_resets, _np_random) -> Line:
    # agent 0 enters via switch A from the north and targets beyond switch B (east), same as before.
    # agents 1, 2 and 3 queue up north of switch B, all sharing agent 1's route: via switch B, the
    # trunk and switch A, to beyond switch A (west).
    # agents 0 and 1 (the ones that actually face each other across the trunk) run at half speed, not
    # full speed -- see _facing_agents_line_generator and
    # test_two_agents_enter_facing_switches_and_deadlock_there's docstring for why. Agents 2 and 3 stay
    # at full speed: they're brought to a halt by physically queueing behind the agent ahead of them (a
    # denied crossing at a cell already occupied, banking distance at the boundary) -- a mechanism
    # independent of, and unaffected by, flatland-rl#178's STOP_MOVING-completes-an-in-flight-crossing
    # fix -- so their final position doesn't depend on their speed the way agents 0/1's does, even
    # though DeadLockAvoidancePolicy also independently issues them STOP_MOVING once it detects agent 0
    # as oncoming on their own path too (see the docstring below).
    return Line(
        agent_waypoints={
            0: [[Waypoint(Q_N_OF_A, int(Grid4TransitionsEnum.NORTH))], target_waypoints(rail, Q_EAST_OF_B)],
            1: [[Waypoint(Q_FIRST, int(Grid4TransitionsEnum.SOUTH))], target_waypoints(rail, Q_WEST_OF_A)],
            2: [[Waypoint(Q_SECOND, int(Grid4TransitionsEnum.SOUTH))], target_waypoints(rail, Q_WEST_OF_A)],
            3: [[Waypoint(Q_THIRD, int(Grid4TransitionsEnum.SOUTH))], target_waypoints(rail, Q_WEST_OF_A)],
        },
        agent_speeds=[0.5, 0.5, 1.0, 1.0],
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

    action_dict, dones = _run_n_steps(env, policy, num_steps=20)

    assert all(action == RailEnvActions.STOP_MOVING for action in action_dict.values()), \
        "expected DeadLockAvoidancePolicy to stop all four agents directly, not just the two facing each other"

    positions = [agent.current_entry_point[0] if agent.current_entry_point is not None else None for agent in env.agents]
    states = [agent.state for agent in env.agents]
    assert positions == [Q_SWITCH_A, Q_SWITCH_B, Q_FIRST, Q_SECOND], \
        "expected agents 0 and 1 to be stuck at their own switch cell, and agents 2/3 one cell behind the agent ahead of them"
    assert states == [TrainState.STOPPED] * 4, "expected all four agents to be permanently stopped"
    assert not dones["__all__"], "expected the agents to deadlock and never all arrive"


# Same infrastructure as above, but with an extra switch AA north of switch A, and targets chosen so
# that only agent 0 and agent 1 share the conflicting segment north of A, while agents 2 and 3's own
# paths stop short of it (at switch A itself):
#
#                    AA_N_OF_AA                                                     AA_Q_THIRD  (agent 3)
#                       |                                                              |
#  AA_WEST_OF_AA -- [AA: simple_switch_north_left]                                AA_Q_SECOND (agent 2)
#                       |                                                              |
#                       |                                                        AA_Q_FIRST  (agent 1)
#                       v                                                              v
#      AA_WEST_OF_A -- [A: simple_switch_west_right] -- trunk (3 cells) -- [B: simple_switch_east_left] -- AA_EAST_OF_B
#
# Agent 0 starts at AA_N_OF_AA and targets AA_EAST_OF_B (unchanged end-to-end route): AA_N_OF_AA -> AA
# (forced south) -> A (forced east, entering from the north) -> trunk -> B -> AA_EAST_OF_B.
# Agent 1 starts at AA_Q_FIRST and targets AA_WEST_OF_AA: AA_Q_FIRST -> B (forced west) -> trunk -> A
# (branches north, since its target lies beyond AA) -> AA (branches west) -> AA_WEST_OF_AA. Its path
# therefore shares the entire AA<->A<->trunk<->B segment with agent 0, in the opposite direction.
# Agents 2 and 3 start further back in the same queue and target AA_WEST_OF_A: their path also goes
# via B and the trunk, but branches west already at switch A, so it never reaches AA at all.
#
# NOTE: flatland-rl#178's "let STOP_MOVING complete an in-flight cell-boundary crossing"
# (178-agents-living-on-the-edge-9) initially appeared to break this scenario's distinction (agent 0's
# already-in-flight crossing would land it one cell further, on switch A itself, which agents 2/3's
# own path also passes through). Widening the rail with a one-cell buffer between AA and switch A, and
# separately bumping DeadLockAvoidancePolicy's min_free_cell from 1 to 2, were both tried to compensate
# and did NOT work: DeadLockAvoidancePolicy._check_agent_can_move only applies min_free_cell once an
# opposing agent is already registered on the shared path (`len_opp_agents == 0: return True`), so it
# is not a static distance-based threshold, and no amount of upstream buffer changes when agent 0 and
# agent 1 become mutually visible. What actually resolves it: agents 0 and 1 run at half speed (see
# _agents_with_aa_line_generator), so their crossing is never "in flight" (pre_offset + pre_speed <
# SEGMENT_LENGTH) when STOP_MOVING is issued -- see
# test_two_agents_enter_facing_switches_and_deadlock_there's docstring for the general mechanism.
AA_N_OF_AA = (2, 1)
AA_AA = (3, 1)
AA_WEST_OF_AA = (3, 0)
AA_SWITCH_A = (4, 1)
AA_WEST_OF_A = (4, 0)
AA_TRUNK = [(4, 2), (4, 3), (4, 4)]
AA_SWITCH_B = (4, 5)
AA_EAST_OF_B = (4, 6)
AA_Q_FIRST = (3, 5)
AA_Q_SECOND = (2, 5)
AA_Q_THIRD = (1, 5)


def _make_two_switches_with_aa_rail() -> RailGridTransitionMap:
    dead_end_from_south = RailEnvTransitionsEnum.dead_end_from_south
    dead_end_from_west = RailEnvTransitionsEnum.dead_end_from_west
    dead_end_from_east = RailEnvTransitionsEnum.dead_end_from_east
    vertical_straight = RailEnvTransitionsEnum.vertical_straight
    horizontal_straight = RailEnvTransitionsEnum.horizontal_straight
    simple_switch_west_right = RailEnvTransitionsEnum.simple_switch_west_right
    simple_switch_east_left = RailEnvTransitionsEnum.simple_switch_east_left
    simple_switch_north_left = RailEnvTransitionsEnum.simple_switch_north_left

    row0 = [0, 0, 0, 0, 0, dead_end_from_south, 0]
    row1 = [0, 0, 0, 0, 0, vertical_straight, 0]
    row2 = [0, dead_end_from_south, 0, 0, 0, vertical_straight, 0]
    row3 = [dead_end_from_east, simple_switch_north_left, 0, 0, 0, vertical_straight, 0]
    row4 = [dead_end_from_east, simple_switch_west_right] + [horizontal_straight] * len(AA_TRUNK) + [simple_switch_east_left, dead_end_from_west]
    return build_rail([row0, row1, row2, row3, row4])


def _agents_with_aa_line_generator(rail, _num_agents, _hints, _num_resets, _np_random) -> Line:
    # agents 0 and 1 (the ones that actually share the AA<->A<->trunk<->B segment) run at half speed,
    # not full speed -- see the NOTE above and _facing_agents_line_generator for why.
    return Line(
        agent_waypoints={
            0: [[Waypoint(AA_N_OF_AA, int(Grid4TransitionsEnum.NORTH))], target_waypoints(rail, AA_EAST_OF_B)],
            1: [[Waypoint(AA_Q_FIRST, int(Grid4TransitionsEnum.SOUTH))], target_waypoints(rail, AA_WEST_OF_AA)],
            2: [[Waypoint(AA_Q_SECOND, int(Grid4TransitionsEnum.SOUTH))], target_waypoints(rail, AA_WEST_OF_A)],
            3: [[Waypoint(AA_Q_THIRD, int(Grid4TransitionsEnum.SOUTH))], target_waypoints(rail, AA_WEST_OF_A)],
        },
        agent_speeds=[0.5, 0.5, 1.0, 1.0],
    )


def _build_env_with_aa() -> RailEnv:
    rail = _make_two_switches_with_aa_rail()
    env = RailEnv(
        width=rail.width,
        height=rail.height,
        rail_generator=rail_from_grid_transition_map(rail),
        line_generator=_agents_with_aa_line_generator,
        # all agents have earliest departure 0, so they enter and reach their switch/queue as early as possible.
        timetable_generator=ttgen_flatland2,
        number_of_agents=4,
        obs_builder_object=FullEnvObservation(),
    )
    env.reset()
    return env


def test_agents_short_of_the_conflict_keep_receiving_move_forward():
    """
    Same queuing setup as above, but this time only agent 0 and agent 1 actually share the
    conflicting segment: agent 0's route now goes via a second switch AA (north of switch A) before
    reaching switch A itself, and agent 1's target is moved out beyond AA too (`AA_WEST_OF_AA`), so
    its path also extends through switch A and AA -- the same segment agent 0 travels, in the
    opposite direction. Agents 2 and 3, in contrast, still target `AA_WEST_OF_A` (short of AA): their
    own path only reaches switch A and never includes the segment north of it.

    Once agent 0 reaches switch AA, it becomes visible (an oncoming train on its own path) to agent 1
    -- which is the only agent whose path extends that far -- and DeadLockAvoidancePolicy stops agent 0
    and agent 1 directly. Agents 2 and 3 never see agent 0 on their path (it is beyond where their own
    route ever goes) and keep receiving MOVE_FORWARD from DeadLockAvoidancePolicy indefinitely; it is
    only the environment's own motion check that keeps them from advancing, since each is blocked by
    the (stationary) agent immediately ahead of it in the queue.
    """
    env = _build_env_with_aa()
    policy = DeadLockAvoidancePolicy(use_entering_prevention=False, min_free_cell=1)

    # Agents 2 and 3 are only held back by the environment's own motion check (queueing behind a
    # stopped agent ahead of them, never receiving STOP_MOVING from DLA itself), which oscillates their
    # state MOVING/STOPPED every single step even once their position has settled -- so the exact
    # iteration count matters: it must land on a STOPPED phase, not a MOVING one.
    action_dict, dones = _run_n_steps(env, policy, num_steps=21)

    assert action_dict[0] == RailEnvActions.STOP_MOVING, "expected agent 0 to be stopped directly by DeadLockAvoidancePolicy"
    assert action_dict[1] == RailEnvActions.STOP_MOVING, "expected agent 1 to be stopped directly by DeadLockAvoidancePolicy"
    assert action_dict[2] == RailEnvActions.MOVE_FORWARD, "expected agent 2 to keep receiving MOVE_FORWARD -- it never sees agent 0 on its own (shorter) path"
    assert action_dict[3] == RailEnvActions.MOVE_FORWARD, "expected agent 3 to keep receiving MOVE_FORWARD -- it never sees agent 0 on its own (shorter) path"

    positions = [agent.current_entry_point[0] if agent.current_entry_point is not None else None for agent in env.agents]
    states = [agent.state for agent in env.agents]
    assert positions == [AA_AA, AA_SWITCH_B, AA_Q_FIRST, AA_Q_SECOND], "expected agents 2 and 3 to be physically queued one cell behind where they started"
    assert states == [TrainState.STOPPED] * 4, "expected agents 2 and 3 to be stopped by the environment's motion check despite DLA issuing MOVE_FORWARD"
    assert not dones["__all__"], "expected the agents to deadlock and never all arrive"
