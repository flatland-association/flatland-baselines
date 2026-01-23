from typing import List, Dict, Tuple

from flatland.envs.agent_chains import AgentHandle
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState


# TODO https://github.com/flatland-association/flatland-baselines/issues/24 backport to flatland-rl after refactorings. We need to re-generate the regression trajectories with `get_k_shortest_paths` instead of custom `ShortestDistanceWalker`. For now use `ShortestDistanceWalker` as regression tests are based on the shortest paths produced by this method.
class DupShortestPathPolicy(RailEnvPolicy[RailEnv, RailEnv, RailEnvActions]):
    """
    Works with `FullEnvObservation`
    """

    def __init__(self, _get_k_shortest_paths=None):
        super().__init__()
        self._shortest_paths: Dict[AgentHandle, Tuple[Waypoint]] = {}

    def _act(self, env: RailEnv, agent: EnvAgent):
        if agent.position is None:
            return RailEnvActions.MOVE_FORWARD

        if len(self._shortest_paths[agent.handle]) == 0:
            return RailEnvActions.DO_NOTHING

        for a in {RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT}:
            new_cell_valid, (new_position, new_direction), transition_valid, preprocessed_action = env.rail.check_action_on_agent(
                RailEnvActions.from_value(a), (agent.position, agent.direction)
            )
            if new_cell_valid and transition_valid and (
                    new_position == self._shortest_paths[agent.handle][1].position and new_direction == self._shortest_paths[agent.handle][1].direction):
                return a
        raise Exception("Invalid state")

    def act_many(self, handles: List[int], observations: List[RailEnv], **kwargs):
        actions = {}
        for handle, env in zip(handles, observations):
            agent = env.agents[handle]
            self._update_agent(agent, env)
            actions[handle] = self._act(env, agent)
        return actions

    def _update_agent(self, agent: EnvAgent, env: RailEnv):
        """
        Build `_shortest_paths`.
        """
        if agent.state == TrainState.DONE:
            self._shortest_paths.pop(agent.handle, None)
            return

        if agent.handle not in self._shortest_paths:
            always_first_waypoint = [pp[0] for pp in agent.waypoints]
            self._set_shortest_path_from_non_flexible_waypoints(agent, always_first_waypoint, env.rail)

        if agent.position is None:
            return

        while self._shortest_paths[agent.handle][0].position != agent.position:
            self._shortest_paths[agent.handle] = self._shortest_paths[agent.handle][1:]
        assert self._shortest_paths[agent.handle][0].position == agent.position

    def _set_shortest_path_from_non_flexible_waypoints(self, agent: EnvAgent, waypoints: List[Waypoint], rail: RailGridTransitionMap):
        """
        Sets the shortest path to path built by routing shortest path between waypoints.

        Assumes the shortest path complies with the directions at the intermediate waypoints.
        """
        p: List[Waypoint] = []
        for p1, p2 in zip(waypoints, waypoints[1:]):
            if len(p) > 0:
                assert p[-1] == p1, (p[-1], p1)
            # TODO add caching for get_k_shortest_paths?
            path_segment_candidates: List[Tuple[Waypoint]] = get_k_shortest_paths(None, p1.position, p1.direction, p2.position, rail=rail)
            next_path_segment = None
            if p2.direction is None:
                next_path_segment = path_segment_candidates[0]
            else:
                # TODO add optional direction at target to get_k_shortest_paths
                for _p_next in path_segment_candidates:
                    if _p_next[-1].direction == p2.direction:
                        next_path_segment = _p_next
                        break
            assert next_path_segment is not None, f"Not found next path from {p1} to {p2}"
            if len(p) > 0:
                p += next_path_segment[1:]
            else:
                p += next_path_segment
        self._shortest_paths[agent.handle] = p
