from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from flatland.core.env_observation_builder import AgentHandle, ObservationBuilder
from flatland.envs.rail_env import RailEnv
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_observation_builder_service import DeadlockAvoidanceObservationBuilderService


@dataclass
class DeadlockAvoidanceObservation:
    """
    Vector-based observation for DLA.
    """
    full_shortest_distance_agent_map: np.ndarray  # type=int, dim=(num_agents, height, width)
    shortest_distance_agent_len: np.ndarray  # type=int, dim=(num_agents,)
    shortest_distance_agent_map: np.ndarray  # type=int, dim=(num_agents, height, width)
    opp_agent_map: np.ndarray  # type=bool, dim=(num_agents, num_agents)


class DeadlockAvoidanceObservationBuilder(ObservationBuilder[RailEnv, DeadlockAvoidanceObservation]):
    """
    Flatland `ObservationBuilder` returning the same `DeadlockAvoidanceObservation` observation for all agents, using `DeadlockAvoidanceObservationBuilderService`'s per-step state computation internally.
    """

    def __init__(self, start_step_service: DeadlockAvoidanceObservationBuilderService):
        self.start_step_service = start_step_service
        self._step_state: Optional[DeadlockAvoidanceObservation] = None

    def reset(self):
        self._step_state = None

    def get_many(self, handles: Optional[List[AgentHandle]] = None) -> Dict[AgentHandle, DeadlockAvoidanceObservation]:
        self.start_step_service.start_step()
        self._step_state = self._to_external()
        if handles is None:
            handles = []
        return {h: self._step_state for h in handles}

    def get(self, handle: AgentHandle = 0) -> DeadlockAvoidanceObservation:
        return self._step_state

    def _to_external(self) -> DeadlockAvoidanceObservation:
        service = self.start_step_service
        num_agents = service._rail_env.get_num_agents()
        opp_agent_map = np.zeros((num_agents, num_agents), dtype=bool)
        for h, opp_set in service._state.opp_agent_map.items():
            for opp_a in opp_set:
                opp_agent_map[h, opp_a] = True
        return DeadlockAvoidanceObservation(
            full_shortest_distance_agent_map=service._state.full_shortest_distance_agent_map,
            shortest_distance_agent_len=np.array(
                [service._state.shortest_distance_agent_len[h] for h in range(num_agents)],
                dtype=int,
            ),
            shortest_distance_agent_map=service._state.shortest_distance_agent_map,
            opp_agent_map=opp_agent_map,
        )
