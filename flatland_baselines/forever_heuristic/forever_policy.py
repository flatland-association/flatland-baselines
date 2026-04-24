from typing import Any

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy


class ForeverPolicy(RailEnvPolicy[RailEnv, Any, RailEnvActions]):
    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        while True:
            pass
