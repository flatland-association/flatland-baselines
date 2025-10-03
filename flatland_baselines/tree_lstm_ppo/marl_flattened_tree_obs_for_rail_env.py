from typing import Optional

import numpy as np

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.ml.observations.gym_observation_builder import GymObservationBuilder
from flatland_baselines.tree_lstm_ppo.pytorch_tree_lstm.example_usage import convert_tree_to_tensors


class MarlFlattenedTreeObsForRailEnv(GymObservationBuilder[RailEnv, np.ndarray], TreeObsForRailEnv):

    def get(self, handle: Optional[AgentHandle] = 0) -> np.ndarray:
        root_node = super().get(handle)
        return convert_tree_to_tensors(root_node)
