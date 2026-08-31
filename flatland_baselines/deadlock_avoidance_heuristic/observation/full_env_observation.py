from flatland.core.env_observation_builder import ObservationBuilder, AgentHandle, ObservationT
from flatland.envs.rail_env import RailEnv


class FullEnvObservation(ObservationBuilder[RailEnv, RailEnv]):
    """
    Returns full env as observation.
    """

    def __init__(self):
        pass

    def get(self, handle: AgentHandle = 0) -> ObservationT:
        return self.env

    def reset(self, env):
        self.env = env
