from ray.rllib.algorithms import AlgorithmConfig, Algorithm

from flatland.ml.ray.examples.flatland_rollout import do_rollout
from flatland.ml.ray.examples.flatland_training_with_parameter_sharing import _get_algo_config_parameter_sharing
from flatland.ml.ray.flatland_metrics_and_trajectory_callback import FlatlandMetricsAndTrajectoryCallback
from flatland.ml.ray.wrappers import ray_policy_wrapper_from_rllib_checkpoint
from flatland_baselines.simple_rllib_ppo.simple_ppo_common import get_simple_ppo_config
from flatland_baselines.simple_rllib_ppo.simple_ppo_training import register_flatland_ray_cli_observation_builders

if __name__ == '__main__':
    register_flatland_ray_cli_observation_builders()

    obs_builder_class = "FlattenedNormalizedTreeObsForRailEnv_max_depth_2_50"
    # obs_builder_class = "FlattenedNormalizedTreeObsForRailEnv_max_depth_3_50"
    obs_builder_class = "ShortestDistanceToTargetObservationBuilderGym"
    # obs_builder_class = "FlattenedTreeObsForRailEnv_max_depth_1_50"
    # obs_builder_class = "FastTreeObservationBuilderGym_2"
    obs_builder_class = "UngroupedFlattenedTreeObsForRailEnv_max_depth_2"
    obs_builder_class = "UngroupedFlattenedTreeObsForRailEnv_max_depth_1"
    checkpoint_path = "/Users/che/workspaces/flatland-benchmarks-f3-starterkit/checkpoint_000199"

    num_agents = 1
    evaluate_only = False
    d = get_simple_ppo_config(num_agents=num_agents)

    algo_config: AlgorithmConfig = _get_algo_config_parameter_sharing(**d)
    algo = algo_config.build_algo()
    # TODO seems not work yet - success rate consistently at 0 whereas often > 0 for RLlib way.
    do_rollout(env=algo.env_creator(None), num_episodes_during_inference=5, policy=ray_policy_wrapper_from_rllib_checkpoint(checkpoint_path, algo, "p0"))

    algo_config: AlgorithmConfig = _get_algo_config_parameter_sharing(**d)
    algo_config.callbacks(FlatlandMetricsAndTrajectoryCallback)
    algo: Algorithm = algo_config.build_algo()

    algo.restore(checkpoint_path)
    assert algo.training_iteration > 0
    # TODO terribly slow - why? observations?
    eval_results = algo.evaluate()
    print(eval_results)

    # TODO https://github.com/ray-project/ray/blob/master/rllib/examples/curriculum/curriculum_learning.py custom eval with small, middle and large envs
