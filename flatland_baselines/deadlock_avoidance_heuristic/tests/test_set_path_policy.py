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

# Two switches A and B on one track; the agent comes from the north arm of A, turns onto the track at A and runs
# straight through B to the east:
#
#                N_OF_A
#                  |
#   WEST_OF_A -- [A] -- . -- . -- . -- [B] -- EAST_OF_B
#                                        |
#                                      N_OF_B
N_OF_A = (0, 1)
SWITCH_A = (1, 1)
EAST_OF_B = (1, 6)


def _line_generator(rail, _num_agents, _hints, _num_resets, _np_random) -> Line:
    return Line(
        agent_waypoints={0: [[Waypoint(N_OF_A, int(Grid4TransitionsEnum.NORTH))], target_waypoints(rail, EAST_OF_B)]},
        agent_speeds=[1.0],
    )


def _build_env() -> RailEnv:
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
    )
    env.reset()
    return env


def test_set_path_policy_on_its_own_follows_its_path_to_the_target():
    """
    `SetPathPolicy` on its own, not as `DeadLockAvoidancePolicy`'s base class: the agent turns at switch A onto the
    track and arrives at its target. The policy sends `MOVE_FORWARD` wherever going straight on follows the path (a
    turn there has no valid transition of its own), and a turn only where the path turns.
    """
    env = _build_env()
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
    assert set(turns) <= {(SWITCH_A, int(Grid4TransitionsEnum.SOUTH))}, f"expected a turn at most at switch A, got {turns}"
