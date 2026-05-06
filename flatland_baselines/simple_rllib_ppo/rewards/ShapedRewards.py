from collections import defaultdict

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.grid.distance_map import DistanceMap
from flatland.envs.rewards import Rewards
from flatland.envs.step_utils.env_utils import AgentTransitionData
from flatland.envs.step_utils.states import TrainState


class ShapedRewards(Rewards[float]):
    def __init__(self):
        # TODO make sure rewards are re-initialized upon reset
        self.reward_signal_updated = defaultdict(lambda: 0)

    def step_reward(self, agent: EnvAgent, agent_transition_data: AgentTransitionData, distance_map: DistanceMap, elapsed_steps: int) -> float:
        if elapsed_steps < 5:
            self.reward_signal_updated[agent.handle] = 0
        if self.reward_signal_updated[agent.handle] == 0 and agent.state == TrainState.DONE:
            self.reward_signal_updated[agent.handle] =  1.0
            return 1.0
        else:
            return - 0.001

    def end_of_episode_reward(self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int) -> float:
        if self.reward_signal_updated[agent.handle] == 0 and agent.state != TrainState.DONE:
            self.reward_signal_updated[agent.handle] =  1.0
            return -1.0
        return 0.0

    def cumulate(self, *rewards: float) -> float:
        return sum(rewards)

    def normalize(self, *rewards: float, num_agents: int, max_episode_steps: int) -> float:
        # https://flatland-association.github.io/flatland-book/challenges/flatland3/eval.html
        return sum(rewards) / (max_episode_steps * num_agents) + 1.0

    def empty(self) -> float:
        return 0
