"""
Trains custom torch model on Flatland env in RLlib using single policy learning.
Based on https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/pettingzoo_parameter_sharing.py.
"""
import argparse
import logging
from typing import Union, Optional

import gymnasium as gym
import numpy as np
import ray
import torch.nn as nn
from ray import tune
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils import override
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.utils.typing import ResultDict
from ray.tune.registry import get_trainable_cls, register_env, registry_get_input, register_input

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.observations import TreeObsForRailEnv, Node
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenedNormalizedTreeObsForRailEnv
from flatland.ml.observations.gym_observation_builder import GymObservationBuilder
from flatland.ml.ray.examples.flatland_observation_builders_registry import register_flatland_ray_cli_observation_builders
from flatland.ml.ray.wrappers import ray_env_generator
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


import gymnasium as gym
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv

from flatland.envs.rail_env import RailEnv
from flatland.ml.ray.ray_multi_agent_rail_env import RayMultiAgentWrapper


# TODO backport to flatland-rl
# See also for https://gitlab.aicrowd.com/flatland/neurips2020-flatland-baselines/-/blob/master/train.py?ref_type=heads
class FlatlandMetricsCallback(RLlibCallback):
    """
    Add `normalized_reward` and `percentage_complete` evaluation metrics.
    """

    def on_episode_end(
            self,
            *,
            episode: MultiAgentEpisode,
            env_runner,
            metrics_logger,
            env,
            env_index,
            rl_module,
            **kwargs,
    ) -> None:
        # If we have a vector env, only render the sub-env at index 0.
        if isinstance(env.unwrapped, (gym.vector.VectorEnv, VectorMultiAgentEnv)):
            unwrapped = env.unwrapped.envs[0]
        else:
            unwrapped = env.unwrapped
        while not isinstance(unwrapped, RayMultiAgentWrapper):
            unwrapped = unwrapped.unwrapped
        rail_env: RailEnv = unwrapped._wrap

        rewards_dict = episode.get_rewards(-1)
        episode.get_state()
        episode_done_agents = 0
        for h in rail_env.get_agent_handles():
            if rail_env.dones[h]:
                episode_done_agents += 1

        episode_num_agents = len(rewards_dict)
        assert episode_num_agents == rail_env.get_num_agents()
        assert sum(list(rewards_dict.values())) == sum(list(rail_env.rewards_dict.values()))
        # https://flatland-association.github.io/flatland-book/challenges/flatland3/eval.html
        normalized_reward = sum(list(rewards_dict.values())) / (
                rail_env._max_episode_steps *
                episode_num_agents
        ) + 1

        metrics_logger.log_value(
            "normalized_reward",
            normalized_reward,
        )

        percentage_complete = float(sum([agent.state == 6 for agent in rail_env.agents])) / episode_num_agents
        metrics_logger.log_value(
            "percentage_complete",
            percentage_complete,
        )

        metrics_logger.log_value(
            "max_episode_steps",
            rail_env._max_episode_steps,
        )

        metrics_logger.log_value(
            "elapsed_steps",
            rail_env._elapsed_steps,
        )

        num_malfunctions = sum([agent.malfunction_handler.num_malfunctions for agent in rail_env.agents])
        metrics_logger.log_value(
            "num_malfunctions",
            num_malfunctions,
        )


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


def setup_func():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s")


def add_flatland_training_with_parameter_sharing_args():
    parser = add_rllib_example_script_args(
        default_iters=200,
        default_timesteps=1000000,
        default_reward=0.0,
    )
    parser.set_defaults(
        enable_new_api_stack=True
    )
    parser.add_argument(
        "--train-batch-size-per-learner",
        type=int,
        default=4000,
        help="See https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.training.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig.training",
    )
    parser.add_argument(
        "--obs-builder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        required=False,
        help="The address of the ray cluster to connect to in the form ray://<head_node_ip_address>:10001. Leave empty to start a new cluster. Passed to ray.init(address=...). See https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html ",
    )
    parser.add_argument("--env_var", "-e",
                        metavar="KEY=VALUE",
                        nargs='*',
                        help="Set ray runtime environment variables like -e RAY_DEBUG=1, passed to ray.init(runtime_env={env_vars: {...}}), see https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#api-reference")
    return parser


def train(args: Optional[argparse.Namespace] = None, init_args=None) -> Union[ResultDict, tune.result_grid.ResultGrid]:
    if args is None:
        parser = add_flatland_training_with_parameter_sharing_args()
        args = parser.parse_args()
    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"
    assert (
        args.obs_builder
    ), "Must set --obs-builder <obs builder ID> when running this script!"

    setup_func()
    if init_args is None:
        env_vars = set()
        if args.env_var is not None:
            env_vars = args.env_var
        init_args = {
            # https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments
            "runtime_env": {
                "env_vars": dict(map(lambda s: s.split('='), env_vars)),
                # https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html
                "worker_process_setup_hook": "flatland.ml.ray.examples.flatland_training_with_parameter_sharing.setup_func"
            },
            "ignore_reinit_error": True,
        }
        if args.ray_address is not None:
            init_args['address'] = args.ray_address

    # https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html
    ray.init(
        **init_args,
    )
    env_name = "flatland_env"
    # test_id,env_id,n_agents,x_dim,y_dim,n_cities,max_rail_pairs_in_city,n_envs_run,grid_mode,max_rails_between_cities,malfunction_duration_min,malfunction_duration_max,malfunction_interval,speed_ratios,random_seed
    # Test_3,Level_0,50,30,35,3,2,10,False,2,20,50,4500,"{1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25}",8524649404651236810
    # x_dim=30,
    register_env(env_name, lambda _: ray_env_generator(
        n_agents=args.num_agents,
        obs_builder_object=registry_get_input(args.obs_builder)(),
        x_dim=30,
        y_dim=35,
        n_cities=3,
        max_rail_pairs_in_city=2,
        grid_mode=False,
        max_rails_between_cities=2,
        malfunction_duration_min=20,
        malfunction_duration_max=50,
        malfunction_interval=540,
        speed_ratios={1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25},
        seed=int(np.random.default_rng().integers(2 ** 32 - 1)),
    ))

    # TODO could be extracted to cli - keep it low key as illustration only
    additional_training_config = {}
    if args.algo == "DQN":
        additional_training_config = {"replay_buffer_config": {
            "type": "MultiAgentEpisodeReplayBuffer",
        }}
    base_config = (
        # N.B. the warning `passive_env_checker.py:164: UserWarning: WARN: The obs returned by the `reset()` method was expecting numpy array dtype to be float32, actual type: float64`
        #   comes from ray.tune.registry._register_all() -->  import ray.rllib.algorithms.dreamerv3 as dreamerv3!
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment("flatland_env")
        .multi_agent(
            policies={"p0"},
            # All agents map to the exact same policy.
            policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
        )
        .training(
            model={
                # "vf_share_layers": True,
            },
            train_batch_size=args.train_batch_size_per_learner,
            **additional_training_config
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={"p0": RLModuleSpec(
                    module_class=SelfAttentionTorchRLModule,
                    model_config={
                        "hidden_sz": 128,
                        "tree_embedding_sz": registry_get_input(args.obs_builder)().get_observation_space().shape[0],
                        "action_sz": 5
                    }
                )},
            )
        )
        .callbacks(FlatlandMetricsCallback)
        .evaluation(
            evaluation_num_env_runners=2,
            evaluation_interval=1,
            evaluation_force_reset_envs_before_iteration=True,
            evaluation_duration=20,
            evaluation_parallel_to_training=False,
            evaluation_duration_unit="episodes"
        )
    )
    res = run_rllib_example_script_experiment(base_config, args)

    if res.num_errors > 0:
        raise AssertionError(f"{res.errors}")
    return res


if __name__ == '__main__':
    register_flatland_ray_cli_observation_builders()

    register_input("UngroupedFlattenedTreeObsForRailEnv_max_depth_3_50",
                   lambda: UngroupedFlattenedTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))
    register_input("FlattenedNormalizedTreeObsForRailEnv_max_depth_2_50",
                   lambda: FlattenedNormalizedTreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))
    parser = add_flatland_training_with_parameter_sharing_args()

    train(parser.parse_args([
        "--num-agents", "50",
        "--obs-builder", "FlattenedNormalizedTreeObsForRailEnv_max_depth_2_50",
        "--algo", "PPO",
        # "--evaluation-num-env-runners", "1",
        # "--evaluation-interval", "1",
        "--checkpoint-freq", "1",
        "--train-batch-size-per-learner", "500",
        # "--stop-iters", "2",
    ]))
