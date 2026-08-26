from flatland.env_generation.env_generator import env_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland_baselines.tree_lstm_ppo.marl_flattened_tree_obs_for_rail_env import MarlFlattenedTreeObsForRailEnv


def test_marl_flattened_tree_obs_for_rail_env():
    obs_builder = MarlFlattenedTreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(max_depth=10))
    env_generator(n_agents=7, obs_builder_object=obs_builder)
    obs = obs_builder.get(2)
    print(obs)
