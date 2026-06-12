from typing import Dict, List, Optional

from flatland.core.env_observation_builder import AgentHandle, ObservationBuilder
from flatland.envs.rail_env import RailEnv
from flatland_baselines.deadlock_avoidance_heuristic.policy.start_step_service import StartStepService, StepStateExternal


class StartStepObservationBuilder(ObservationBuilder[RailEnv, StepStateExternal]):
    def __init__(self, start_step_service: StartStepService):
        self.start_step_service = start_step_service
        self._step_state: Optional[StepStateExternal] = None

    def reset(self):
        self._step_state = None

    def get_many(self, handles: Optional[List[AgentHandle]] = None) -> Dict[AgentHandle, StepStateExternal]:
        self._step_state = self.start_step_service.start_step()
        if handles is None:
            handles = []
        return {h: self._step_state for h in handles}

    def get(self, handle: AgentHandle = 0) -> StepStateExternal:
        return self._step_state
