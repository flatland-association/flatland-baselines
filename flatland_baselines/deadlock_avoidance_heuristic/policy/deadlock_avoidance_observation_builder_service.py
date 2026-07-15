from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Set, Tuple, Optional, Callable

import matplotlib.pyplot as plt
import numpy as np

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.fast_methods import fast_count_nonzero
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState


@dataclass
class DeadlockAvoidanceInternalObservationBuilderState:
    # start_step (1): -1 if no agent, agent handle otherwise
    agent_positions: np.ndarray
    # start_step (2.2): 1 if current shortest path (without current cell!), 0 otherwise
    full_shortest_distance_agent_map: np.ndarray
    # start_step (2.2): all positions on current shortest path (without current cell!)
    shortest_distance_positions_agent_map: Dict[AgentHandle, Set[Tuple[int, int]]]
    # start_step (2.2): directions for all positions on current shortest path (without current cell!)
    shortest_distance_positions_directions_agent_map: Dict[AgentHandle, Dict[Tuple[int, int], Set[int]]]
    # start_step (2.3.3): number of cells till first oncoming agent (without current cell!)
    shortest_distance_agent_len: Dict[AgentHandle, int]
    # start_step (2.3.3): 1 if current shortest path (without current cell!) before first oncoming train, 0 else.
    shortest_distance_agent_map: np.ndarray
    # start_step (2.3.2): set of oncoming agents
    opp_agent_map: Dict[AgentHandle, Set[AgentHandle]]


class DeadlockAvoidanceObservationBuilderService:
    """
    Computes/updates, per step, each agent's shortest-path bitmap and oncoming-agent (opposition) state, and decides whether an agent may move without risking a deadlock.
    Separates concerns for observation building
    """

    def __init__(
            self,
            min_free_cell: int,
            count_num_opp_agents_towards_min_free_cell: bool,
            use_switches_heuristic: bool,
            use_entering_prevention: bool,
            show_debug_plot: bool,
            verbose: bool,
            audit: bool,
    ):
        self.min_free_cell = min_free_cell
        self.count_num_opp_agents_towards_min_free_cell = count_num_opp_agents_towards_min_free_cell
        self.use_switches_heuristic = use_switches_heuristic
        self.use_entering_prevention = use_entering_prevention
        self.show_debug_plot = show_debug_plot
        self.verbose = verbose
        self.audit = None
        if audit:
            self.audit = []

        self._rail_env: Optional[RailEnv] = None
        # N.B. shared state between service and policy
        self._set_paths: Optional[Dict[AgentHandle, Tuple[Waypoint]]] = None
        self._update_agent: Optional[Callable[[EnvAgent, RailEnv], None]] = None
        self._switches: Optional[np.ndarray] = None
        self._state: Optional[DeadlockAvoidanceInternalObservationBuilderState] = None

    def init_env(self, rail_env: RailEnv, set_paths: Dict[AgentHandle, Tuple[Waypoint]], update_agent_fn: Callable[[EnvAgent, RailEnv], None]) -> None:
        self._rail_env = rail_env
        self._set_paths = set_paths
        self._update_agent = update_agent_fn

        self._switches = None
        if self.use_switches_heuristic:
            self._switches = np.zeros((rail_env.height, rail_env.width), dtype=int)
            for r in range(rail_env.height):
                for c in range(rail_env.width):
                    if self._is_switch_cell((r, c)):
                        self._switches[(r, c)] = 1

        num_agents = rail_env.get_num_agents()
        self._state = DeadlockAvoidanceInternalObservationBuilderState(
            agent_positions=np.zeros((rail_env.height, rail_env.width), dtype=int) - 1,
            full_shortest_distance_agent_map=np.zeros((num_agents, rail_env.height, rail_env.width), dtype=int),
            shortest_distance_positions_agent_map=defaultdict(set),
            shortest_distance_positions_directions_agent_map=defaultdict(lambda: defaultdict(set)),
            shortest_distance_agent_len=defaultdict(lambda: 0),
            shortest_distance_agent_map=np.zeros((num_agents, rail_env.height, rail_env.width), dtype=int),
            opp_agent_map=defaultdict(set),
        )

    def start_step(self) -> None:
        # (1)
        self._build_agent_position_map()
        # (2)
        self._update_shortest_distance_maps_and_opp_agent_map()

    def init_shortest_distance_positions(self, agent: EnvAgent, handle: AgentHandle) -> None:
        """
        start_step (2.2.1) / act -> find_alternative()

        Initializes:
        - `_state.full_shortest_distance_agent_map`
        - `_state.shortest_distance_positions_agent_map`
        - `_state.shortest_distance_positions_directions_agent_map`
        """
        self._state.full_shortest_distance_agent_map[handle].fill(0)
        self._state.shortest_distance_positions_agent_map[handle] = set()
        self._state.shortest_distance_positions_directions_agent_map[handle] = defaultdict(set)
        if self._set_paths[agent.handle] is None:
            return
        for wp in self._set_paths[agent.handle][1:]:
            position, direction = wp.position, wp.direction
            self._state.full_shortest_distance_agent_map[(handle, position[0], position[1])] = 1
            self._state.shortest_distance_positions_agent_map[handle].add(position)
            self._state.shortest_distance_positions_directions_agent_map[handle][position].add(direction)

    def invalidate_opposition(self, handle: AgentHandle) -> None:
        """Clears cached opposition tracking for `handle`, forcing it to be recomputed from scratch on the
        next `start_step()` (e.g. after the agent's path changed via rerouting)."""
        self._state.opp_agent_map[handle] = set()

    def _build_agent_position_map(self):
        """
        start_step (1): update agent positions at start of step.
        """
        self._state.agent_positions = np.zeros((self._rail_env.height, self._rail_env.width), dtype=int) - 1
        for handle in range(self._rail_env.get_num_agents()):
            agent = self._rail_env.agents[handle]
            if agent.state in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
                if agent.position is not None:
                    self._state.agent_positions[agent.position] = handle

    def _update_shortest_distance_maps_and_opp_agent_map(self):
        """
        start_step (2): update the shortest paths to current position and update opposing agent
        as well as bitmap representation at start of step.
        """
        # (2.0)
        all_agent_positions: Set[Tuple[int, int]] = self._collect_all_agent_positions()

        for agent in self._rail_env.agents:
            handle = agent.handle

            # (2.1)
            self._update_agent(agent, self._rail_env)

            # (2.2)
            self._build_full_shortest_distance_agent_map(agent, handle)
            if agent.state == TrainState.DONE or agent.state == TrainState.WAITING:
                continue

            # (2.3)
            self._build_shortest_distance_agent_map(agent, handle, all_agent_positions)

    def _collect_all_agent_positions(self) -> Set[Tuple[int, int]]:
        """start_step (2.0)"""
        all_agent_positions = set()
        for agent in self._rail_env.agents:
            all_agent_positions.add(agent.position)
        return all_agent_positions

    def _build_full_shortest_distance_agent_map(self, agent: EnvAgent, handle: AgentHandle):
        """start_step (2.2)"""
        # (2.2.1)
        if self._rail_env._elapsed_steps == 1:
            self.init_shortest_distance_positions(agent, handle)
        # (2.2.2)
        if agent.position is not None and agent.position != agent.old_position:
            assert agent.position == self._set_paths[agent.handle][0].position
            if agent.position not in {wp.position for wp in self._set_paths[agent.handle][1:]}:
                self._state.full_shortest_distance_agent_map[(handle, agent.position[0], agent.position[1])] = 0
                if agent.old_position is not None:
                    self._state.shortest_distance_positions_agent_map[handle].remove(agent.position)
            if agent.old_position is not None:
                self._state.shortest_distance_positions_directions_agent_map[handle][agent.position].remove(int(agent.direction))

    def _build_shortest_distance_agent_map(self, agent: EnvAgent, handle: AgentHandle, all_agent_positions: Set[Tuple[int, int]]):
        """start_step (2.3)"""
        # (2.3.1)
        prev_opp_agents = self._state.opp_agent_map[handle]
        overlap = self._state.shortest_distance_positions_agent_map[handle].intersection(all_agent_positions)
        if overlap == prev_opp_agents:
            return

        # (2.3.2)
        self._rebuild_opp_agent_map(handle, overlap)
        # (2.3.3)
        self._rebuild_shortest_distance_agent_map(agent, handle)

    def _rebuild_opp_agent_map(self, handle: AgentHandle, overlap: Set[Tuple[int, int]]):
        """start_step (2.3.2)"""
        self._state.opp_agent_map[handle] = set()
        for position in overlap:
            opp_a = self._state.agent_positions[position]
            if opp_a != -1 and opp_a != handle:
                directions = self._state.shortest_distance_positions_directions_agent_map[handle][position]
                assert len(directions) > 0, f"Inconsistency for agent {handle} at {self._rail_env._elapsed_steps}: no directions for position {position}"
                for direction in directions:
                    if self._rail_env.agents[opp_a].direction != direction:
                        self._state.opp_agent_map[handle].add(opp_a)

    def _rebuild_shortest_distance_agent_map(self, agent: EnvAgent, handle: AgentHandle):
        """start_step (2.3.3)"""
        self._state.shortest_distance_agent_map[handle].fill(0)
        self._state.shortest_distance_agent_len[handle] = 0
        num_opp_agents = 0
        for wp in self._set_paths[agent.handle][1:]:
            position, direction = wp.position, wp.direction
            opp_a = self._state.agent_positions[position]
            if opp_a != -1 and opp_a != handle:
                if self._rail_env.agents[opp_a].direction != direction:
                    num_opp_agents += 1
                    break
            if num_opp_agents == 0:
                self._state.shortest_distance_agent_len[handle] += 1
                self._state.shortest_distance_agent_map[(handle, position[0], position[1])] = 1

    def _get_action(self, configuration: Tuple[Tuple[int, int], int], next_configuration: Tuple[Tuple[int, int], int]):
        for action in [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]:
            new_cell_valid, new_configuration, transition_valid, preprocessed_action, _ = self._rail_env.rail._check_action_on_agent(action, configuration)
            if new_configuration == next_configuration:
                return preprocessed_action
        raise

    def _check_agent_can_move(
            self,
            my_shortest_walking_path: np.ndarray,
            my_shortest_walking_path_len: int,
            opp_agents: np.ndarray,
            full_shortest_distance_agent_map: np.ndarray,
            handle: AgentHandle,
            switches: Optional[np.ndarray] = None,
            count_num_opp_agents_towards_min_free_cell: bool = False,
            debug: bool = False,
    ):
        """
        The algorithm collects for each train along its route all trains that are currently on a resource in the route.
        For each collected train (`opp_agents`), the method has to decide at which position along the route the train
        must let pass the collected opposing train:  by searching the train's path required resources backward along the path
        starting at the collected train position; stop the search when the resource along the collected train's path is not equal.
        This yields `free_cells` ahead of the agent without overlap with any opposing agent's travelling path.
        If `free_cells >= min_free_cells >= 1` for all opposing agents, then the agent can move.
        A deadlock can only occur if a jam "fills in" the free space and is not detected by the algorithm.

        To determine `free_cells`, the implementation compares takes the difference of
        - the bitmap of the agent's shortest path (up to first opposing agent) and
        - the bitmap opposing agent's path
        and counts the positive elements.

        The forward and backward traveling along the train and the collected train path must be done step-by-step synchronous.
        If the first non-equal resource position along the train's path is more than one resource from train's current location away,
        then the train can move and no deadlock will occur for the next time step.

        2 heuristics to avoid "fill-in":
        - switches: if switches is given, then switches do not count towards free cells
        - count_num_opp_agents_towards_min_free_cell: the number of opposing agents is added to `min_free_cell`
        """
        len_opp_agents = int(np.sum(opp_agents))
        if len_opp_agents == 0:
            return True

        if my_shortest_walking_path_len < self.min_free_cell - len_opp_agents:
            if self.verbose:
                print(f" *** {self._rail_env._elapsed_steps}: agent cannot move")
            return False
        min_free_cell = self.min_free_cell
        if count_num_opp_agents_towards_min_free_cell:
            min_free_cell += len_opp_agents

        for opp_a in np.nonzero(opp_agents)[0]:
            opp = full_shortest_distance_agent_map[opp_a]
            if switches is None:
                free_cells = np.count_nonzero((my_shortest_walking_path - opp) > 0)
            else:
                free_cells = np.count_nonzero((my_shortest_walking_path - switches - opp) > 0)

            if free_cells < min_free_cell:
                free = self._get_free(handle, opp_a)

                if self.verbose:
                    print(
                        f" *** {self._rail_env._elapsed_steps}: agent {handle} blocked by {opp_a} with {free_cells}: {free}. All oncoming agents on path {opp_agents}")
                if self.audit is not None:
                    self.audit.append({"env_time": self._rail_env._elapsed_steps, "agent_id": handle, "k": "audit",
                                       "v": f" *** {self._rail_env._elapsed_steps}: agent {handle} blocked by {opp_a} with {free_cells}: {free}. All oncoming agents on path {opp_agents}"})
                if debug:
                    self._plot_debug(handle, opp_a, my_shortest_walking_path, full_shortest_distance_agent_map, opp, free_cells)

                return False
        return True

    def _plot_debug(
            self,
            handle: AgentHandle,
            opp_a: AgentHandle,
            my_shortest_walking_path: np.ndarray,
            full_shortest_distance_agent_map: np.ndarray,
            opp: np.ndarray,
            free_cells: int,
    ) -> None:
        cells_1 = [wp.position for wp in self._set_paths[handle]]
        cells_2 = [wp.position for wp in self._set_paths[opp_a]]
        if self.verbose:
            print(f"cells_1 = {cells_1}; cells_2={cells_2}")
        im1 = np.zeros((self._rail_env.height, self._rail_env.width))
        for cell in cells_1:
            im1[cell] = 1
        ax = plt.subplot(1, 2, 1)
        ax.set_title(f"Agent {handle} set path ({len(cells_1)})")
        plt.imshow(im1)

        im2 = np.zeros((self._rail_env.height, self._rail_env.width))
        for cell in cells_2:
            im2[cell] = 1
        ax = plt.subplot(1, 2, 2)
        ax.set_title(f"Agent {opp_a} set path ({len(cells_2)})")
        plt.imshow(im2)
        plt.show()

        ax = plt.subplot(4, 1, 1)
        ax.set_title(f"Agent {handle} full path ({np.count_nonzero(full_shortest_distance_agent_map[handle])})")
        plt.imshow(full_shortest_distance_agent_map[handle])

        ax = plt.subplot(4, 1, 2)
        ax.set_title(f"Agent {handle} my_shortest_walking_path ({np.count_nonzero(my_shortest_walking_path)})")
        plt.imshow(my_shortest_walking_path)

        ax = plt.subplot(4, 1, 3)
        ax.set_title(f"Agent {opp_a} full path ({np.count_nonzero(opp)})")
        plt.imshow(opp)

        ax = plt.subplot(4, 1, 4)
        ax.set_title(f"Agent {handle} - agent free_cells  {opp_a} ({free_cells})")
        plt.imshow(my_shortest_walking_path - opp)
        plt.show()

    def _get_free(self, handle: AgentHandle, opp_a: AgentHandle):
        """
        How many cells free ahead of me till other agent.
        Returns zero if no path set for myself.
        Returns full path minus current position if no overlap or no path set for other agent.
        """
        own_path = self._set_paths.get(handle, None)
        opp_path = self._set_paths.get(opp_a, None)
        if own_path is None:
            return 0
        elif opp_path is None:
            return len(own_path) - 1
        return _get_free_from_path(own_path, opp_path)

    @lru_cache(maxsize=100000)
    def _is_switch_cell(self, position) -> bool:
        for new_dir in range(4):
            possible_transitions = self._rail_env.rail.get_transitions((position, new_dir))
            num_transitions = fast_count_nonzero(possible_transitions)
            if num_transitions > 1:
                return True
        return False


def _get_free_from_path(own_path: List[Waypoint], opp_path: List[Waypoint]):
    my_cells = {wp.position for wp in own_path[1:]}
    opp_cells = {wp.position for wp in opp_path}
    my_cells_own = my_cells.difference(opp_cells)
    num = 0
    for i, wp in enumerate(own_path):
        num = i
        if wp.position not in my_cells_own:
            num = i - 1
            break
    free = own_path[:num + 1]
    return free
