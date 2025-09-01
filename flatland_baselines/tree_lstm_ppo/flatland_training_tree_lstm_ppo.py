"""
Runs Flatland env in RLlib using single policy learning, based on
- https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/pettingzoo_parameter_sharing.py.

Take this as starting point to build your own training (cli) script.
"""
import argparse
import logging
import sys
from typing import Union, Optional

import numpy as np
import ray
import torch
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
from ray.tune.registry import get_trainable_cls, register_env, registry_get_input

from flatland.ml.ray.examples.flatland_observation_builders_registry import register_flatland_ray_cli_observation_builders
from flatland.ml.ray.wrappers import ray_env_generator


# https://docs.ray.io/en/latest/rllib/getting-started.html#rllib-python-api
# TODO make configurable/registry?
class CustomTorchRLModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        # You have access here to the following already set attributes:
        # self.observation_space
        # self.action_space
        # self.inference_only
        # self.model_config  # <- a dict with custom settings
        input_dim = self.observation_space.shape[0]
        hidden_dim = 5
        # hidden_dim = self.model_config["hidden_dim"]
        output_dim = self.action_space.n

        # Define and assign your torch subcomponents.
        self._policy_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        self._vl = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Linear(output_dim, 1)
        )

    def _forward(self, batch, **kwargs):
        # Push the observations from the batch through our `self._policy_net`.
        action_logits = self._policy_net(batch[Columns.OBS])
        # Return parameters for the default action distribution, which is
        # `TorchCategorical` (due to our action space being `gym.spaces.Discrete`).
        return {Columns.ACTION_DIST_INPUTS: action_logits}

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        # TODO use value head
        # warnings.warn(str(batch))
        return self._vl(batch["obs"]).squeeze(-1)


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
    #                   y_dim=30,
    #                   n_cities=2,
    #                   max_rail_pairs_in_city=4,
    #                   grid_mode=False,
    #                   max_rails_between_cities=2,
    #                   malfunction_duration_min=20,
    #                   malfunction_duration_max=50,
    #                   malfunction_interval=540,
    #                   speed_ratios=None,
    #                   seed=42,
    register_env(env_name, lambda _: ray_env_generator(
        n_agents=args.num_agents, obs_builder_object=registry_get_input(args.obs_builder)(),
        # TODO inject
        x_dim=50,
        y_dim=30,
        n_cities=35,
        max_rail_pairs_in_city=3,
        grid_mode=False,
        max_rails_between_cities=2,
        malfunction_duration_min=20,
        malfunction_duration_max=50,
        malfunction_interval=4000,
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
            # TODO inject TreeLSTM model
            model={
                # "vf_share_layers": True,
            },
            train_batch_size=args.train_batch_size_per_learner,
            **additional_training_config
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={"p0": RLModuleSpec(
                    # TODO make configurable/use TreeLSTM
                    module_class=CustomTorchRLModule,
                    # model_config={"hidden_dim": 5}
                )},
            )
        )
    )
    res = run_rllib_example_script_experiment(base_config, args)

    if res.num_errors > 0:
        raise AssertionError(f"{res.errors}")
    return res


if __name__ == '__main__':
    register_flatland_ray_cli_observation_builders()
    parser = add_flatland_training_with_parameter_sharing_args()
    # args = parser.parse_args()

    a = int(np.random.default_rng().integers(sys.maxsize))

    train(parser.parse_args([
        "--num-agents", "7",
        "--obs-builder", "FlattenedNormalizedTreeObsForRailEnv_max_depth_3_50",
        "--algo", "PPO",
        # TODO evaluation seems not to work
        "--evaluation-num-env-runners", "1", "--evaluation-interval", "1",
        # TODO load and run Trajectory/metadata.csv ...
        "--checkpoint-freq", "1"
    ]))
    # train(args)
