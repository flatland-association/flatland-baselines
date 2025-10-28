"""
Trains custom torch model on Flatland env in RLlib using single policy learning.
Based on https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/pettingzoo_parameter_sharing.py.
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import torch.nn as nn
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils import override
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.tune.registry import registry_get_input

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.observations import TreeObsForRailEnv, Node
from flatland.envs.rail_env import RailEnv
from flatland.ml.observations.gym_observation_builder import GymObservationBuilder
from flatland.ml.ray.examples.flatland_observation_builders_registry import register_flatland_ray_cli_observation_builders
from flatland.ml.ray.examples.flatland_training_with_parameter_sharing import train_with_parameter_sharing
from flatland_baselines.tree_lstm_ppo.flatland_marl.net_tree import Transformer


# TODO backport to flatland-rl
# from flatland.ml.observations.flatten_tree_observation_for_rail_env import UngroupedFlattenedTreeObsForRailEnv
class UngroupedFlattenedTreeObsForRailEnv(GymObservationBuilder[RailEnv, np.ndarray], TreeObsForRailEnv):
    """
    Gym-ified and flattened normalized tree observation without feature grouping.
    """

    NUM_FEATURES = 12
    NUM_BRANCHES = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_len_data(tree_depth: int, num_features):
        k = num_features
        for _ in range(tree_depth):
            k = k * UngroupedFlattenedTreeObsForRailEnv.NUM_BRANCHES + num_features
        return k

    def _traverse_subtree(self, node, current_tree_depth: int, max_tree_depth: int) -> np.ndarray:
        if node == -np.inf:
            remaining_depth = max_tree_depth - current_tree_depth
            # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
            num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
            return [-np.inf] * num_remaining_nodes * self.NUM_FEATURES

        data = list(node[:UngroupedFlattenedTreeObsForRailEnv.NUM_FEATURES])

        if not node.childs:
            return data

        for direction in TreeObsForRailEnv.tree_explored_actions_char:
            sub_data = self._traverse_subtree(node.childs[direction], current_tree_depth + 1, max_tree_depth)
            data = np.concatenate((data, sub_data))

        return data

    def traverse_tree(self, root_node: Node, max_tree_depth: int) -> np.ndarray:
        """
        This function extracts:

        - `data` (12 features per node):
            - `node.dist_own_target_encountered`
            - `node.dist_other_target_encountered`
            - `node.dist_other_agent_encountered`
            - `node.dist_potential_conflict`
            - `node.dist_unusable_switch`
            - `node.dist_to_next_branch`
            - `node.dist_min_to_target`
            - `node.num_agents_same_direction`
            - `node.num_agents_opposite_direction`
            - `node.num_agents_malfunctioning`
            - `node.speed_min_fractional`
            - `node.num_agents_ready_to_depart`

        Subtrees are traversed depth-first in pre-order (i.e. Node 'N' itself, 'L', 'F', 'R', 'B').
        See `get_len_flattened()` for the length of the flattened structure.
        """
        data = list(root_node[:12])

        for direction in UngroupedFlattenedTreeObsForRailEnv.tree_explored_actions_char:
            sub_data = self._traverse_subtree(root_node.childs[direction], 1, max_tree_depth)
            data = np.concatenate((data, sub_data))

        return data

    def get(self, handle: Optional[AgentHandle] = 0) -> np.ndarray:
        root_node = super(UngroupedFlattenedTreeObsForRailEnv, self).get(handle)
        make_complete_nary(root_node, 0, self.max_depth, UngroupedFlattenedTreeObsForRailEnv.tree_explored_actions_char)
        return self.traverse_tree(root_node, self.max_depth)

    def get_observation_space(self, handle: int = 0) -> gym.Space:
        k = self.get_len_flattened()
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(k,), dtype=np.float64)

    def get_len_flattened(self):
        """
        The total size `S[k]` of the flattened structure for `max_tree_depth=k` is recursively defined by:
        - `S[0] = NUM_FEATURES`
        - `S[k+1] = S[k] * NUM_BRANCHES + NUM_FEATURES`
        for
        - `NUM_FEATURES=12`
        - `NUM_BRANCHES=4`

        I.e.
        - max_depth=1 -> 60
        - max_depth=2 -> 252
        - max_depth=3 -> 1020
        - ...

        Returns
        -------
        Length of the flattened tree obs.
        """

        k = UngroupedFlattenedTreeObsForRailEnv.NUM_FEATURES
        for _ in range(self.max_depth):
            k = k * UngroupedFlattenedTreeObsForRailEnv.NUM_BRANCHES + UngroupedFlattenedTreeObsForRailEnv.NUM_FEATURES
        return k


def unflatten(flat, index, depth, max_depth, num_features, tree_explored_actions_char):
    if (flat[index * num_features:index * num_features + num_features] == -np.inf).all():
        # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
        sub_tree_size = int((4 ** (max_depth - depth + 1) - 1) / (4 - 1))
        return -np.inf, index + sub_tree_size
    n = Node(*flat[index * num_features:index * num_features + num_features], childs={})
    index += 1
    if depth < max_depth:
        for a in tree_explored_actions_char:
            child, index = unflatten(flat, index, depth + 1, max_depth, num_features, tree_explored_actions_char)
            n.childs[a] = child
    return n, index


def make_complete_nary(node: Node, depth, max_depth, tree_explored_actions_char):
    if depth < max_depth:
        for a in tree_explored_actions_char:
            if a not in node.childs or node.childs[a] == -np.inf:
                node.childs[a] = Node(*[-np.inf] * 12, {})
            make_complete_nary(node.childs[a], depth + 1, max_depth, tree_explored_actions_char)


# https://docs.ray.io/en/latest/rllib/getting-started.html#rllib-python-api
# TODO make configurable/registry?
class SelfAttentionTorchRLModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        # You have access here to the following already set attributes:
        # self.observation_space
        # self.action_space
        # self.inference_only
        # self.model_config  # <- a dict with custom settings

        # Define and assign your torch subcomponents.
        # aka. "Self-Attention Block"
        hidden_sz = self.model_config["hidden_sz"]
        tree_embedding_sz = self.model_config["tree_embedding_sz"]
        action_sz = self.model_config["action_sz"]
        self.transformer = nn.Sequential(
            nn.Linear(tree_embedding_sz, tree_embedding_sz),
            nn.GELU(),
            nn.Linear(tree_embedding_sz, tree_embedding_sz),
            nn.GELU(),
            nn.Linear(tree_embedding_sz, tree_embedding_sz),
            Transformer(tree_embedding_sz, 4),
            Transformer(tree_embedding_sz, 4),
            Transformer(tree_embedding_sz, 4),
        )

        # N.B. in contrast to paper, the linear block is part of Action Head and Value Head here (i.e. trained separately), whereas in paper it's part of Self-Attention Block

        # aka. "Action Head"
        self.actor_net = nn.Sequential(
            nn.Linear(tree_embedding_sz, hidden_sz),
            nn.GELU(),
            nn.Linear(hidden_sz, hidden_sz),
            nn.GELU(),
            nn.Linear(hidden_sz, action_sz),
        )

        # aka. "Value Head"
        self.critic_net = nn.Sequential(
            nn.Linear(tree_embedding_sz, hidden_sz),
            nn.GELU(),
            nn.Linear(hidden_sz, hidden_sz),
            nn.GELU(),
            nn.Linear(hidden_sz, 1),
        )

    def _forward(self, batch, **kwargs):
        # torch.Size([7, 1020])
        _batch = batch[Columns.OBS]
        _batch = _batch.unsqueeze(0)
        embedding = self.transformer.forward(_batch)
        action_logits = self.actor_net.forward(embedding).squeeze(0)

        # Return parameters for the default action distribution, which is
        # `TorchCategorical` (due to our action space being `gym.spaces.Discrete`).
        return {Columns.ACTION_DIST_INPUTS: action_logits}

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        # TODO use embeddings to avoid redoing forward pass
        _batch = batch[Columns.OBS]
        _batch = _batch.unsqueeze(0)
        embedding = self.transformer.forward(_batch)
        # Squeeze out last dimension (single node value head).
        return self.critic_net.forward(embedding).squeeze(-1).squeeze(0)


if __name__ == '__main__':
    register_flatland_ray_cli_observation_builders()
    obs_builder_class = "FlattenedNormalizedTreeObsForRailEnv_max_depth_2_50"

    parser = add_rllib_example_script_args()
    train_with_parameter_sharing(
        # args going into run_rllib_example_script_experiment()
        args=parser.parse_args([
            "--enable-new-api-stack",
            "--algo", "PPO",
            "--evaluation-interval", "1",
            "--checkpoint-freq", "1",
            "--num-agents", "50",
        ]),
        callbacks_pkg="flatland.ml.ray.flatland_metrics_callback",
        callbacks_cls="FlatlandMetricsCallback",
        train_batch_size_per_learner=500,
        module_class=SelfAttentionTorchRLModule,
        obs_builder_class=registry_get_input(obs_builder_class),
        model_config={
            "hidden_sz": 128,
            "tree_embedding_sz": registry_get_input(obs_builder_class)().get_observation_space().shape[0],
            "action_sz": 5
        },
        # test_id,env_id,n_agents,x_dim,y_dim,n_cities,max_rail_pairs_in_city,n_envs_run,seed,grid_mode,max_rails_between_cities,malfunction_duration_min,malfunction_duration_max,malfunction_interval,speed_ratios
        # Test_03,Level_0,50,30,35,3,2,10,42,False,2,20,50,540,"{1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}"
        # see https://flatland-association.github.io/flatland-book/challenges/flatland3/envconfig.html Round 2, Test03
        env_config=dict(
            x_dim=30,
            y_dim=35,
            n_cities=3,
            max_rail_pairs_in_city=2,
            grid_mode=False,
            max_rails_between_cities=2,
            malfunction_duration_min=20,
            malfunction_duration_max=50,
            malfunction_interval=540,
            speed_ratios={1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}, ),
    )

    # TODO custom eval with small, middle and large envs
    # TODO test wandb logging
