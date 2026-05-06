from ray.rllib.examples.utils import add_rllib_example_script_args
from ray.tune.registry import registry_get_input, register_input

from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.ml.observations.flatten_tree_observation_for_rail_env import FlattenedTreeObsForRailEnv, FlattenedNormalizedTreeObsForRailEnv
from flatland.ml.observations.gym_observation_builder import DummyObservationBuilderGym, GlobalObsForRailEnvGym
from flatland_baselines.simple_rllib_ppo.models.simple_actor_critic import SimpleActorCriticTorchRLModule
from flatland_baselines.simple_rllib_ppo.observations.fast_tree_obs import FastTreeObservationBuilderGym
from flatland_baselines.simple_rllib_ppo.observations.flatland_fast_tree_observation import FlatlandFastTreeObservationBuilderGym
from flatland_baselines.simple_rllib_ppo.observations.shortest_distance import ShortestDistanceToTargetObservationBuilderGym
from flatland_baselines.simple_rllib_ppo.observations.ungrouped_flattened_tree_obs import UngroupedFlattenedTreeObsForRailEnv
from flatland_baselines.simple_rllib_ppo.rewards.ShapedRewards import ShapedRewards


def register_flatland_ray_cli_observation_builders():
    register_input("DummyObservationBuilderGym", lambda: DummyObservationBuilderGym()),
    register_input("GlobalObsForRailEnvGym", lambda: GlobalObsForRailEnvGym()),
    register_input("FlattenedTreeObsForRailEnv_max_depth_1_50",
                   lambda: FlattenedTreeObsForRailEnv(max_depth=1, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))
    register_input("FlattenedNormalizedTreeObsForRailEnv_max_depth_2_50",
                   lambda: FlattenedNormalizedTreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))
    register_input("FlattenedNormalizedTreeObsForRailEnv_max_depth_2_50",
                   lambda: FlattenedNormalizedTreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))
    register_input("FlattenedNormalizedTreeObsForRailEnv_max_depth_3_50",
                   lambda: FlattenedNormalizedTreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv(max_depth=50)))
    register_input("ShortestDistanceToTargetObservationBuilderGym", lambda: ShortestDistanceToTargetObservationBuilderGym()),
    register_input("FastTreeObservationBuilderGym_3", lambda: FastTreeObservationBuilderGym(3)),
    register_input("FastTreeObservationBuilderGym_2", lambda: FastTreeObservationBuilderGym(2)),
    register_input("FlatlandFastTreeObservationBuilderGym", lambda: FlatlandFastTreeObservationBuilderGym()),
    register_input("UngroupedFlattenedTreeObsForRailEnv_max_depth_2", lambda: UngroupedFlattenedTreeObsForRailEnv(max_depth=2)),
    register_input("UngroupedFlattenedTreeObsForRailEnv_max_depth_1", lambda: UngroupedFlattenedTreeObsForRailEnv(max_depth=1)),


def get_simple_ppo_config(
        num_agents:int,
        obs_builder_class:str
) -> dict:
    parser = add_rllib_example_script_args()
    d = dict(
        # args going into run_rllib_example_script_experiment()
        args=parser.parse_args([
            # "--enable-new-api-stack",
            # "--algo", "DQN",
            "--algo", "PPO",
            "--evaluation-interval", "5",
            # "--checkpoint-freq", "1",
            "--num-agents", f"{num_agents}",
            "--stop-iters", "500",
            "--stop-timesteps", "200000",
            # For debugging, use the following additional command line options
            # "--no-tune",
            # "--num-env-runners=0"
            # which should allow you to set breakpoints anywhere in the RLlib code and
            # have the execution stop there for inspection and debugging
        ]),
        callbacks_pkg="flatland.ml.ray.flatland_metrics_callback",
        callbacks_cls="FlatlandMetricsCallback",
        train_batch_size_per_learner=1024,
        obs_builder_class=registry_get_input(obs_builder_class),
        module_class=SimpleActorCriticTorchRLModule,
        # module_class=None,
        model_config={
            "hidden_sz": 256,
            "state_sz": registry_get_input(obs_builder_class)().get_observation_space().shape[0],
            "action_sz": 5,
            # "vf_share_layers": True,
        },

        env_config=dict(
            max_rails_between_cities=2,
            max_rail_pairs_in_city=4,
            malfunction_interval=1000,
            x_dim=30,
            y_dim=40,
            grid_mode=True,
            # speed_ratios={1.0: 0.25, 0.5: 0.25, 0.33: 0.25, 0.25: 0.25},
            rewards=ShapedRewards(),
        ),
        additional_training_config={
            # 'lr': 0.5e-3,
            # # 'clip_param':0.2,
            # # "model": {
            # #     "vf_loss_coeff": True,
            # # },
            # 'vf_loss_coeff': 0.05
        },

    )
    return d
