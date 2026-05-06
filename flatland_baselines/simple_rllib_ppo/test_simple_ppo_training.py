import time

from flatland_baselines.simple_rllib_ppo.flatland_ray_examples.flatland_training_with_parameter_sharing import train_with_parameter_sharing
from flatland_baselines.simple_rllib_ppo.simple_ppo_common import get_simple_ppo_config, register_flatland_ray_cli_observation_builders


def test_simple_ppo_training():
    register_flatland_ray_cli_observation_builders()
    start_time = time.time()
    config = get_simple_ppo_config(num_agents=1, obs_builder_class="ShortestDistanceToTargetObservationBuilderGym")
    config["args"].stop_timesteps = 1000
    res = train_with_parameter_sharing(**config)
    end_time = time.time()
    print(f"Trainings took {end_time - start_time:.2f} seconds")
    print(res)