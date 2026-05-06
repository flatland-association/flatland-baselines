"""
Trains custom torch model on Flatland env in RLlib using single policy learning.
Based on https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/pettingzoo_parameter_sharing.py.
"""
import time

from flatland_baselines.simple_rllib_ppo.flatland_ray_examples.flatland_training_with_parameter_sharing import train_with_parameter_sharing
from flatland_baselines.simple_rllib_ppo.simple_ppo_common import get_simple_ppo_config, register_flatland_ray_cli_observation_builders


def simple_ppo_training(
        num_agents=1,
        obs_builder_class="ShortestDistanceToTargetObservationBuilderGym",
        # obs_builder_class = "FlattenedNormalizedTreeObsForRailEnv_max_depth_2_50"
        # obs_builder_class = "FlattenedNormalizedTreeObsForRailEnv_max_depth_3_50"

        # obs_builder_class = "FlattenedTreeObsForRailEnv_max_depth_1_50"
        # obs_builder_class = "FastTreeObservationBuilderGym_2"
        # obs_builder_class = "UngroupedFlattenedTreeObsForRailEnv_max_depth_2"
        # obs_builder_class = "UngroupedFlattenedTreeObsForRailEnv_max_depth_1"

):
    register_flatland_ray_cli_observation_builders()
    start_time = time.time()
    config = get_simple_ppo_config(num_agents=num_agents, obs_builder_class=obs_builder_class)
    res = train_with_parameter_sharing(**config)
    end_time = time.time()
    print(f"Trainings took {end_time - start_time:.2f} seconds")
    print(res)

    # TODO curriculum: https://github.com/ray-project/ray/blob/master/rllib/examples/curriculum/curriculum_learning.py
    # TODO test wandb logging


if __name__ == '__main__':
    simple_ppo_training()
