import pytest
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.grid.rail_env_grid import RailEnvTransitionsEnum
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.timetable_generators import ttgen_flatland2
from flatland.envs.timetable_utils import Line
from flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation import FullEnvObservation
from flatland_baselines.deadlock_avoidance_heuristic.policy.set_path_policy import SetPathPolicy
from flatland_baselines.deadlock_avoidance_heuristic.tests.rail_test_utils import build_rail, target_waypoints

# Two switches A and B on one track; the agent comes from the north arm of A, is turned onto the track at A (the only
# transition from that arm), and at B either goes straight on to EAST_OF_B or turns left to N_OF_B (a real choice):
#
#                N_OF_A                      N_OF_B
#                  |                           |
#   WEST_OF_A -- [A] -- . -- . -- . -- [B] -- EAST_OF_B
N_OF_A = (0, 1)
SWITCH_A = (1, 1)
N_OF_B = (0, 5)
SWITCH_B = (1, 5)
EAST_OF_B = (1, 6)


def _build_env(remove_agents_at_target: bool = True, target=EAST_OF_B) -> RailEnv:
    def _line_generator(rail, _num_agents, _hints, _num_resets, _np_random) -> Line:
        return Line(
            agent_waypoints={0: [[Waypoint(N_OF_A, int(Grid4TransitionsEnum.NORTH))], target_waypoints(rail, target)]},
            agent_speeds=[1.0],
        )

    row0 = [0, RailEnvTransitionsEnum.dead_end_from_south, 0, 0, 0, RailEnvTransitionsEnum.dead_end_from_south, 0]
    row1 = [RailEnvTransitionsEnum.dead_end_from_east, RailEnvTransitionsEnum.simple_switch_west_right] \
           + [RailEnvTransitionsEnum.horizontal_straight] * 3 \
           + [RailEnvTransitionsEnum.simple_switch_east_left, RailEnvTransitionsEnum.dead_end_from_west]
    rail = build_rail([row0, row1])
    env = RailEnv(
        width=rail.width,
        height=rail.height,
        rail_generator=rail_from_grid_transition_map(rail),
        line_generator=_line_generator,
        timetable_generator=ttgen_flatland2,
        number_of_agents=1,
        obs_builder_object=FullEnvObservation(),
        remove_agents_at_target=remove_agents_at_target,
    )
    env.reset()
    return env


TURN_AT_A = {(SWITCH_A, int(Grid4TransitionsEnum.SOUTH)): RailEnvActions.MOVE_LEFT}


@pytest.mark.parametrize("target,expected_turns", [
    (EAST_OF_B, TURN_AT_A),
    (N_OF_B, {**TURN_AT_A, (SWITCH_B, int(Grid4TransitionsEnum.EAST)): RailEnvActions.MOVE_LEFT}),
])
def test_set_path_policy_on_its_own_follows_its_path_to_the_target(target, expected_turns):
    """
    `SetPathPolicy` on its own, not as `DeadLockAvoidancePolicy`'s base class: the agent arrives at its target. The
    policy sends `MOVE_FORWARD` where the path goes straight on, `MOVE_LEFT` for the turn to the east at A (the only
    transition from the north arm), and at B, where the agent has a choice, `MOVE_LEFT` only for the target N_OF_B.
    """
    env = _build_env(target=target)
    policy = SetPathPolicy()
    observations = env._get_observations()
    agent = env.agents[0]

    actions_at = {}
    dones = {"__all__": False}
    for _ in range(20):
        lookahead_from = agent.next_entry_point
        action_dict = policy.act_many(env.get_agent_handles(), observations=list(observations.values()))
        if agent.current_entry_point is not None:
            actions_at[lookahead_from] = action_dict[0]
        observations, _, dones, _ = env.step(action_dict)
        if dones["__all__"]:
            break

    assert dones["__all__"], "expected the agent to arrive at its target"
    assert agent.arrival_time is not None
    turns = {entry_point: action for entry_point, action in actions_at.items() if action != RailEnvActions.MOVE_FORWARD}
    assert turns == expected_turns


def test_set_path_policy_on_its_own_acts_for_an_agent_parked_at_its_target():
    """
    With `remove_agents_at_target=False`, the agent stays on the map at its target once it arrives. `SetPathPolicy` drops
    its set path then, and must still give it an action in the steps after its arrival, instead of looking the path up.
    """
    env = _build_env(remove_agents_at_target=False)
    policy = SetPathPolicy()
    observations = env._get_observations()
    agent = env.agents[0]

    for _ in range(20):
        action_dict = policy.act_many(env.get_agent_handles(), observations=list(observations.values()))
        if agent.target_entry_point is not None:
            break
        observations, _, _, _ = env.step(action_dict)
    else:
        raise AssertionError("expected the agent to arrive at its target")

    assert agent.current_entry_point is not None, "expected the agent to stay on the map at its target"
    assert action_dict[0] == RailEnvActions.DO_NOTHING


def test_set_path_policy_on_its_own_writes_the_audit_log():
    """`SetPathPolicy` on its own writes its audit entries, with the env's step count, once its audit log is on."""
    env = _build_env()
    policy = SetPathPolicy()
    policy.audit = []

    policy.act_many(env.get_agent_handles(), observations=list(env._get_observations().values()))

    assert policy.audit == [{"env_time": 0, "agent_id": 0, "k": "audit",
                             "v": f"get path for agent 0 ignoring intermediate stops on {env.agents[0].waypoints}"}]
