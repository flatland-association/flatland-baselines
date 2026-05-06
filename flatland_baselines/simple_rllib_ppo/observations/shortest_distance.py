import gymnasium as gym
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState
from flatland.ml.observations.gym_observation_builder import GymObservationBuilder


class ShortestDistanceToTargetObservationBuilderGym(GymObservationBuilder[RailEnv, np.ndarray]):
    """
    Return the shortest distance to target for each action.
    """

    def get_observation_space(self, handle: int = 0) -> gym.Space:
        return self.observation_space

    def __init__(self, clipping=1000):
        super().__init__()
        self.clipping = clipping
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(5,), dtype=float)

    def get(self, handle: int = 0):
        agent = self.env.agents[handle]
        obs = np.zeros((5,))
        if agent.state != TrainState.DONE:
            for action in RailEnvActions:
                rc = self.env.rail.apply_action_independent(action, agent.current_configuration or agent.initial_configuration)
                if rc is not None:
                    next_configuration, _ = rc
                    obs[action.value] = self.env.distance_map._get_distance(next_configuration, agent.handle)
        return np.clip(obs, 0, self.clipping)

    def reset(self):
        pass

