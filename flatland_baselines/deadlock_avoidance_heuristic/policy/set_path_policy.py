import warnings
from collections import defaultdict
from functools import lru_cache
from typing import Any, List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from flatland.envs.agent_chains import AgentHandle
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState


class SetPathPolicy(RailEnvPolicy[RailEnv, RailEnv, RailEnvActions]):
    """
    Works with `FullEnvObservation` only.

    Policy where agents follow a set path.
    """

    def __init__(self,
                 k_shortest_path_cutoff: int = None,
                 use_always_first_strategy: int = None,
                 verbose: bool = False,
                 ):
        """

        Parameters
        ----------
        k_shortest_path_cutoff : Optional[int]
            Global cutoff for shortest path finding. Use with care as it can cause agents to have no set path at all.
        use_always_first_strategy : Optional[int]
            If not set or zero, ignore intermediate stops altogether. Otherwise, use shortest path between first option at all intermediate stops.
        verbose : bool
        """
        super().__init__()
        self._set_paths: Dict[AgentHandle, Tuple[Waypoint]] = {}
        self.k_shortest_path_cutoff = k_shortest_path_cutoff
        self.use_alternative_at_first_intermediate_and_then_always_first_strategy = use_always_first_strategy
        self.verbose = verbose
        # audit log, off (None) by default; turned on by DeadLockAvoidancePolicy(audit=True) or by setting it to []
        self.audit: Optional[List[Dict]] = None
        # step count of the env last seen in `_update_agent`, for the audit log
        self._elapsed_steps: Optional[int] = None

    def _act(self, env: RailEnv, agent: EnvAgent):
        if agent.current_entry_point is None:
            return RailEnvActions.MOVE_FORWARD

        # done: `_update_agent` has dropped the set path, also for an agent parked at its target (`remove_agents_at_target=False`)
        if agent.handle not in self._set_paths or len(self._set_paths[agent.handle]) == 0:
            return RailEnvActions.DO_NOTHING

        # design: actions applied at cell entry -- once agent.next_entry_point holds a
        # pending target (decided one cell ago), this step's action no longer decides that
        # transition (already settled) but the look-ahead beyond it, so it must be evaluated from
        # next_entry_point against the set-path waypoint after it, not from current_entry_point
        # against the immediate next waypoint.
        has_pending_target = agent.next_entry_point is not None and agent.next_entry_point != agent.current_entry_point
        lookahead_from = agent.next_entry_point if has_pending_target else agent.current_entry_point
        target_index = 2 if has_pending_target else 1
        if target_index >= len(self._set_paths[agent.handle]):
            # nothing left to look ahead into; the pending target is already the final waypoint,
            # so permission to continue toward it doesn't depend on this step's action.
            return RailEnvActions.MOVE_FORWARD

        # a list, not a set: where several actions lead onto the path (e.g. MOVE_FORWARD and the turn into the only
        # transition of a switch arm), the first one wins, and a set's order changes with the hash seed of the process
        for a in [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]:
            new_cell_valid, (new_position, new_direction), transition_valid, _, _ = env.rail._check_action_on_agent(
                RailEnvActions.from_value(a), lookahead_from
            )
            if new_cell_valid and transition_valid and (
                    new_position == self._set_paths[agent.handle][target_index].position
                    and new_direction == self._set_paths[agent.handle][target_index].direction):
                return a
        raise Exception("Invalid state")

    def _env_time(self) -> Optional[int]:
        """The env's step count for the audit log: of the env last seen in `_update_agent`."""
        return self._elapsed_steps

    def _audit(self, agent_id: Any, message: str) -> None:
        """Append `message` to the audit log, if the audit is on."""
        if self.audit is not None:
            self.audit.append({"env_time": self._env_time(), "agent_id": agent_id, "k": "audit", "v": message})

    def _log(self, agent_id: Any, message: str) -> None:
        """Print `message` if verbose, and append it to the audit log (`_audit`)."""
        if self.verbose:
            print(message)
        self._audit(agent_id, message)

    @staticmethod
    def _derived_state(agent: EnvAgent, env: RailEnv) -> TrainState:
        """
        The agent's state, derived by flatland-rl from its position, speed, malfunction and earliest departure and the
        env's step count (`EnvAgent.derived_state`), instead of `agent.state`. Read between two steps, it equals the
        state the state machine tracks, and it is also right when the env does not run the state machine
        (`RailEnvStateMachineWrapper` absent or skipping its updates), where `agent.state` never changes.
        """
        return agent.derived_state(env._elapsed_steps)

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
        self._elapsed_steps = env._elapsed_steps
        if self._derived_state(agent, env) == TrainState.DONE:
            self._set_paths.pop(agent.handle, None)
            return

        if agent.handle not in self._set_paths:
            if self.use_alternative_at_first_intermediate_and_then_always_first_strategy is not None and self.use_alternative_at_first_intermediate_and_then_always_first_strategy > 0:
                always_first_waypoint = _always_first_waypoint_from_flexible_groups(agent.waypoints)
                self._log(agent.handle, f"get path for agent {agent.handle} using always-first strategy on {agent.waypoints}")
                self._set_paths[agent.handle] = self._shortest_path_from_non_flexible_waypoints(always_first_waypoint, env.rail,
                                                                                                debug_label=f"Agent {agent.handle}")
            else:
                self._log(agent.handle, f"get path for agent {agent.handle} ignoring intermediate stops on {agent.waypoints}")
                self._set_paths[agent.handle] = self._shortest_path_from_non_flexible_waypoints(
                    _always_first_waypoint_from_flexible_groups([agent.waypoints[0], agent.waypoints[-1]]), env.rail,
                    debug_label=f"Agent {agent.handle}")
        if self._set_paths[agent.handle] is None:
            # loopy path
            return
        if agent.current_entry_point is None:
            # not on map
            return

        position = agent.current_entry_point[0]
        while len(self._set_paths[agent.handle]) > 0 and self._set_paths[agent.handle][0].position != position:
            self._set_paths[agent.handle] = self._set_paths[agent.handle][1:]
        assert self._set_paths[agent.handle][0].position == position

    def _shortest_path_from_non_flexible_waypoints(self, waypoint_groups: List[List[Waypoint]], rail: RailGridTransitionMap, debug_segments: bool = False,
                                                   debug_label: str = ""):
        """
        Computes the shortest path to path built by routing the shortest path between non-flexible waypoints; only target may have flexibility.
        """
        p: List[Waypoint] = []
        for g1, g2 in zip(waypoint_groups, waypoint_groups[1:]):
            # non-target
            assert len(g1) == 1
            p1 = g1[0]
            if len(p) > 0:
                assert p[-1] == p1, (p[-1], p1)

            arrival_directions = {wp.direction for wp in g2}
            target_direction = next(iter(arrival_directions)) if len(arrival_directions) == 1 else None

            path_segment_candidates: List[Tuple[Waypoint]] = _get_k_shortest_paths(None, p1.position, p1.direction, g2[0].position, rail=rail,
                                                                                   target_direction=target_direction, cutoff=self.k_shortest_path_cutoff)
            assert len(path_segment_candidates) > 0, \
                f"[{debug_label}] Not found next path from {p1} to any of {g2}. Either not connected or no path respecting k_shortest_path_cutoff={self.k_shortest_path_cutoff}."
            next_path_segment = path_segment_candidates[0]
            assert g2[0].position == next_path_segment[-1].position
            assert len(set(next_path_segment)) == len(next_path_segment)
            if None not in arrival_directions:
                assert next_path_segment[-1].direction in arrival_directions, \
                    f"[{debug_label}] Not found next path from {p1} to any of {g2}. Either not connected or no path respecting k_shortest_path_cutoff={self.k_shortest_path_cutoff}."
            if len(p) > 0:
                p += next_path_segment[1:]
            else:
                p += next_path_segment
            if debug_segments:
                print(f"Segment {p1} {g2} has len {len(next_path_segment)}")
            if debug_segments:
                cells = [wp.position for wp in next_path_segment]
                height = max([r for (r, c) in cells]) + 1
                width = max([c for (r, c) in cells]) + 1
                data = np.zeros(shape=(height, width))
                for cell in cells:
                    data[cell] = 1
                plt.imshow(data)
                plt.show()
        if len(set(p)) != len(p):
            # TODO dla currently cannot handle this - generalize dla implementation?
            counts = defaultdict(lambda: 0)
            for wp in p:
                counts[wp] += 1
            duplicates = {wp for wp, count in counts.items() if count > 1}

            warnings.warn(f"[{debug_label}] Found loopy line {waypoint_groups} with duplicates {duplicates} in path {p}")
            self._audit(debug_label, f"[{debug_label}] Found loopy line {waypoint_groups} with duplicates {duplicates} in path {p}")
            return None
        return p


@lru_cache
def _get_k_shortest_paths(*args, **kwargs):
    return get_k_shortest_paths(*args, **kwargs)


def _always_first_waypoint_from_flexible_groups(waypoint_groups: List[List[Waypoint]]) -> List[List[Waypoint]]:
    """
    Reduces a list of waypoint-alternatives groups to a single, non-flexible path plan for
    `_shortest_path_from_non_flexible_waypoints`: every intermediate group is narrowed to its first
    alternative (arbitrary but stable), while the *final* group is passed through in
    full.
    """
    if len(waypoint_groups) == 0:
        return []
    return [[pp[0]] for pp in waypoint_groups[:-1]] + [waypoint_groups[-1]]
