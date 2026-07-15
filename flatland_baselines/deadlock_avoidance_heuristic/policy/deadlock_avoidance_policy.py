import copy
import os
import warnings
from collections import defaultdict
from typing import List, Any, Dict, Set, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState
from flatland_baselines.deadlock_avoidance_heuristic.policy.set_path_policy import SetPathPolicy, _get_k_shortest_paths
from flatland_baselines.deadlock_avoidance_heuristic.policy.start_step_service import StartStepService, StepStateExternal

# LRU cache infrastructure kept for backwards compatibility (no functions registered here after refactoring)
flatland_deadlock_avoidance_policy_lru_cache_functions = []


def _enable_flatland_deadlock_avoidance_policy_lru_cache(*args, **kwargs):
    from functools import lru_cache

    def decorator(func):
        func = lru_cache(*args, **kwargs)(func)
        flatland_deadlock_avoidance_policy_lru_cache_functions.append(func)
        return func

    return decorator


def _send_flatland_deadlock_avoidance_policy_data_change_signal_to_reset_lru_cache():
    for func in flatland_deadlock_avoidance_policy_lru_cache_functions:
        func.cache_clear()


class DeadLockAvoidancePolicy(SetPathPolicy):
    def __init__(self,
                 min_free_cell: int = 1,
                 show_debug_plot: bool = False,
                 count_num_opp_agents_towards_min_free_cell: bool = True,
                 use_switches_heuristic: bool = True,
                 use_entering_prevention: bool = False,
                 use_alternative_at_first_intermediate_and_then_always_first_strategy: int = None,
                 drop_next_threshold: int = None,
                 k_shortest_path_cutoff: int = None,
                 seed: int = None,
                 verbose: bool = False,
                 audit: bool = False,
                 ):
        """

        Parameters
        ----------
        min_free_cell : int
            How many cells must be left empty ahead to avoid collisions.
        show_debug_plot : bool
            Show plots with set path and own path till first opposing agent for all agents.
        count_num_opp_agents_towards_min_free_cell : bool
            Subtract the number of other trains oncoming on my own path from the number of free cells ahead of me.
        use_switches_heuristic : bool
            Subtract the number of switches on my own path from the number of free cells ahead of me.
        use_entering_prevention : bool
            Prevent one of two agents entering at the same timestep if `min_free_cells` is not respected.
            No alternative paths, only set path considered for detection. Agents always enter with initial set path.
        use_alternative_at_first_intermediate_and_then_always_first_strategy : Optional[int]
            If set and non-zero, use `use_always_first_strategy` initially for initial set path.
            If entered and blocked by DLA, sample from `k` shortest paths for all options at the next intermediate stop.
        drop_next_threshold : Optional[int]
            When threshold of time steps blocked consecutively, drop the next intermediate stop for the agent. Done iteratively.
        k_shortest_path_cutoff : Optional[int]
            Global cutoff for shortest path finding. Use with care as it can cause agents to have no set path at all.
        seed : Optional[int]
            Seed for sampling of altneratives.
        verbose :bool
        audit :bool
        """
        super().__init__(
            k_shortest_path_cutoff=k_shortest_path_cutoff,
            use_always_first_strategy=use_alternative_at_first_intermediate_and_then_always_first_strategy is not None and use_alternative_at_first_intermediate_and_then_always_first_strategy > 0,
            verbose=verbose,
        )

        self.loss = 0
        self.show_debug_plot = show_debug_plot
        self.min_free_cell = min_free_cell
        self.count_num_opp_agents_towards_min_free_cell = count_num_opp_agents_towards_min_free_cell
        self.use_switches_heuristic = use_switches_heuristic
        self.use_entering_prevention = use_entering_prevention
        self.use_k_alternatives_at_first_intermediate_and_then_always_first_strategy = use_alternative_at_first_intermediate_and_then_always_first_strategy
        self.drop_next_threshold = drop_next_threshold
        self.k_shortest_path_cutoff = k_shortest_path_cutoff

        # will be injected from observation (`FullEnvObservation`)
        self.rail_env: Optional[RailEnv] = None

        self.agent_waypoints_done: Dict[AgentHandle, Set[Waypoint]] = defaultdict(set)
        self.agent_waypoints_tried: Dict[AgentHandle, Set[str]] = defaultdict(set)

        self.closed = defaultdict(list)
        self.num_blocked = defaultdict(lambda: 0)
        self.alternatives = defaultdict(list)

        self.np_random = np.random.RandomState(seed)
        self.audit = None
        if audit is True:
            self.audit = []

        # start_step (3): next (r,c,d) and action to get there; or no entry if train must not move
        self.agent_can_move: Dict[AgentHandle, Tuple[int, int, int, RailEnvActions]] = {}

        self.start_step_service: Optional[StartStepService] = None
        self.step_state: Optional[StepStateExternal] = None

    def _init_env(self, env: RailEnv):
        self.start_step_service = StartStepService(
            min_free_cell=self.min_free_cell,
            count_num_opp_agents_towards_min_free_cell=self.count_num_opp_agents_towards_min_free_cell,
            use_switches_heuristic=self.use_switches_heuristic,
            use_entering_prevention=self.use_entering_prevention,
            show_debug_plot=self.show_debug_plot,
            verbose=self.verbose,
            audit=self.audit is not None,
        )
        self.start_step_service.init_env(
            rail_env=self.rail_env,
            # N.B. state coupling!
            set_paths=self._set_paths,
            update_agent_fn=super()._update_agent,
        )

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        assert isinstance(observations[0], RailEnv)
        if self.rail_env is None:
            self.rail_env = observations[0]
            self._init_env(self.rail_env)
        self.step_state = self.start_step_service.start_step()
        self._extract_agent_can_move()
        return {handle: self._act(handle, observations[handle]) for handle in handles}

    def _act(self, handle: int, state, eps=0.) -> RailEnvActions:
        check = self.agent_can_move.get(handle, None)
        agent = self.rail_env.agents[handle]
        if (agent.handle not in self._set_paths or self._set_paths[agent.handle] is None) and agent.state < TrainState.MOVING:
            # prevent entering map as default!
            act = RailEnvActions.DO_NOTHING
        else:
            act = RailEnvActions.STOP_MOVING

        if agent.position is not None:
            self.agent_waypoints_done[handle].add(Waypoint(agent.position, agent.direction))

        if check is not None:
            act = check[3]
            self.num_blocked[handle] = 0
        else:
            self.num_blocked[handle] += 1
            if agent.state in [TrainState.MOVING, TrainState.STOPPED]:
                self._find_alternative(agent)

        # TODO port to client.py:  File "msgpack/_packer.pyx", line 257, in msgpack._cmsgpack.Packer._pack_inner
        # submission-1      | TypeError: can not serialize 'RailEnvActions' object
        # if isinstance(act, RailEnvActions):
        #    act = act.value
        return act

    def _extract_agent_can_move(self):
        """start_step (3): update whether agent can move."""
        self.agent_can_move = {}

        for handle in range(self.rail_env.get_num_agents()):
            agent = self.rail_env.agents[handle]
            if TrainState.DONE > agent.state >= TrainState.WAITING:
                if self.start_step_service._check_agent_can_move(
                        self.step_state.shortest_distance_agent_map[handle],
                        self.step_state.shortest_distance_agent_len[handle],
                        self.step_state.opp_agent_map[handle],
                        self.step_state.full_shortest_distance_agent_map,
                        agent.handle,
                        self.start_step_service._switches,
                        self.count_num_opp_agents_towards_min_free_cell,
                ):
                    if agent.position is not None:
                        position = agent.position
                        direction = agent.direction
                    else:
                        position = agent.initial_position
                        direction = agent.initial_direction
                    if self._set_paths[agent.handle] is None or len(self._set_paths[agent.handle]) < 2:
                        warnings.warn(f"No shortest path for agent {agent.handle}. Found: {self._set_paths[agent.handle]}")
                        if self.audit is not None:
                            self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": handle, "k": "audit",
                                               "v": f"No shortest path for agent {agent.handle}. Found: {self._set_paths[agent.handle]}"})
                        continue
                    next_position = self._set_paths[agent.handle][1].position
                    next_direction = self._set_paths[agent.handle][1].direction
                    action = self.start_step_service._get_action((position, direction), (next_position, next_direction))

                    self.agent_can_move.update({handle: [next_position[0], next_position[1], next_direction, action]})

        if self.use_entering_prevention:
            entering_agents = [handle for handle, agent in enumerate(self.rail_env.agents) if
                               agent.state == TrainState.READY_TO_DEPART and self.agent_can_move.get(handle, None)]
            if len(entering_agents) > 0:
                if self.verbose:
                    print(f" ++++ {self.rail_env._elapsed_steps} entering {entering_agents}")
                for a1 in entering_agents:
                    for a2 in entering_agents:
                        if a1 != a2 and a1 in self.agent_can_move and a2 in self.agent_can_move:
                            free = self.start_step_service._get_free(a1, a2)
                            if len(free) < self.min_free_cell:
                                self.agent_can_move.pop(a1)
                                if self.verbose:
                                    print(f"!!!! prevent entering conflict {a1, a2} -> let not enter {a1}")

        if self.show_debug_plot:
            a = np.floor(np.sqrt(self.rail_env.get_num_agents()))
            b = np.ceil(self.rail_env.get_num_agents() / a)
            for handle in range(self.rail_env.get_num_agents()):
                plt.subplot(a, b, handle + 1)
                plt.imshow(self.step_state.full_shortest_distance_agent_map[handle] + self.step_state.shortest_distance_agent_map[handle])
            plt.show(block=False)
            plt.pause(0.01)

    def _log(self, agent_handle: AgentHandle, message: str) -> None:
        """Prints `message` if verbose, and records it to the audit trail if auditing is enabled."""
        if self.verbose:
            print(message)
        if self.audit is not None:
            self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": agent_handle, "k": "audit", "v": message})

    def _find_alternative(self, agent: EnvAgent):
        handle = agent.handle
        self._log(handle, f"considering {handle} at {self.rail_env._elapsed_steps}: {self._set_paths[handle]}")

        # TODO optimization: instead of computing the remaining flexible waypoints, update the list on the go. No priority for now
        remaining_flexible_waypoints = self._get_remaining_flexible_waypoints(agent)
        if self.drop_next_threshold is not None and self.num_blocked[handle] > self.drop_next_threshold and len(remaining_flexible_waypoints) > 1:
            self._log(handle, f"dropping next intermediate for {agent.handle} at {self.rail_env._elapsed_steps}, blocked for {self.num_blocked[agent.handle]}")
            remaining_flexible_waypoints = remaining_flexible_waypoints[1:]

        if self.use_k_alternatives_at_first_intermediate_and_then_always_first_strategy is not None and \
                self.use_k_alternatives_at_first_intermediate_and_then_always_first_strategy > 0 and \
                len(remaining_flexible_waypoints[0]) > 0:
            before = self._set_paths[handle]

            if handle not in self.alternatives or self.alternatives[handle][0][0] != Waypoint(agent.position, agent.direction):
                self._log(handle, f"need to re-compute for agent {handle} at {agent.position, agent.direction} at {self.rail_env._elapsed_steps}")

                alternatives = []
                for first_intermediate in remaining_flexible_waypoints[0]:
                    then_always_first_intermediates = [first_intermediate] + [pp[0] for pp in remaining_flexible_waypoints[1:]]
                    prefixes = _get_k_shortest_paths(None, agent.position, agent.direction, first_intermediate.position,
                                                     target_direction=first_intermediate.direction,
                                                     rail=self.rail_env.rail,
                                                     k=self.use_k_alternatives_at_first_intermediate_and_then_always_first_strategy,
                                                     cutoff=self.k_shortest_path_cutoff)
                    suffix = self._shortest_path_from_non_flexible_waypoints(then_always_first_intermediates, self.rail_env.rail,
                                                                             debug_label=f"Agent {agent.handle}")
                    for prefix in prefixes:
                        alternatives.append(list(prefix) + suffix[1:])
                self.alternatives[handle] = alternatives

            # as in set path before:
            self.closed[handle].append(before)

            # randomize the alternative if all alternatives already tried
            assert len(self.alternatives[handle]) > 0, "Either cutoff too low or not reachable."
            alternative = self.alternatives[handle][self.np_random.randint(len(self.alternatives[handle]))]
            for alt in self.alternatives[handle]:
                if alt not in self.closed[handle]:
                    alternative = alt
            self.closed[handle].append(alternative)

            self._log(handle, f"get new path for agent {handle} using alternative-at-first-intermediate-and-then-always-first strategy on {agent.waypoints}")

            before_len = len(before) if before is not None else None
            after_len = len(self._set_paths[handle]) if self._set_paths[handle] is not None else None
            if before == self._set_paths[handle]:
                self._log(handle,
                          f"not changed {handle} at {self.rail_env._elapsed_steps} {before_len}->{after_len}:\n - {before} \n - {self._set_paths[handle]}")
            else:
                self._log(handle, f"changed {handle} at {self.rail_env._elapsed_steps} {before_len}->{after_len}:\n - {before} \n - {self._set_paths[handle]}")

            if self._set_paths[handle] is None or len(self._set_paths[handle]) == 0:
                self._set_paths[handle] = before
            self.start_step_service.init_shortest_distance_positions(agent, handle)
            self.step_state.opp_agent_map[handle] = False

    def _get_remaining_flexible_waypoints(self, agent):
        remaining_flexible_waypoints: List[List[Waypoint]] = copy.deepcopy(agent.waypoints)
        while True:
            if set(remaining_flexible_waypoints[0]).isdisjoint(self.agent_waypoints_done[agent.handle]):
                break
            remaining_flexible_waypoints = remaining_flexible_waypoints[1:]
        assert len(remaining_flexible_waypoints) > 0
        return remaining_flexible_waypoints

    def save(self, filename):
        pass

    def load(self, filename):
        pass


class DeadlockAvoidanceHeuristics(DeadLockAvoidancePolicy):
    def __init__(self,
                 use_alternative_at_first_intermediate_and_then_always_first_strategy=2,
                 seed: int = None,
                 audit: bool = False,
                 ):
        seed = os.environ.get("DLA_SEED", seed)
        if seed is not None:
            seed = int(seed)
        super().__init__(
            count_num_opp_agents_towards_min_free_cell=False,
            use_switches_heuristic=False,
            use_entering_prevention=True,
            use_alternative_at_first_intermediate_and_then_always_first_strategy=use_alternative_at_first_intermediate_and_then_always_first_strategy,
            drop_next_threshold=20,
            k_shortest_path_cutoff=450,
            seed=seed,
            audit=audit,
        )
