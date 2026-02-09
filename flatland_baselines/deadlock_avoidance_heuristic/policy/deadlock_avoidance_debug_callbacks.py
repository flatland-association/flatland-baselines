from pathlib import Path
from typing import Optional, List

import pandas as pd

from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.envs.rail_env import RailEnv
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy


class DeadlockAvoidanceDebugCallbacks(FlatlandCallbacks[RailEnv]):
    def __init__(self, policy: DeadLockAvoidancePolicy):
        self._policy = policy
        self.dfs: List[dict] = []

    def on_episode_step(
            self,
            *,
            env: Optional[RailEnv] = None,
            data_dir: Path = None,
            **kwargs,
    ) -> None:
        self.dfs += [{
            "env_time": env._elapsed_steps, "agent_id": agent.handle, "k": "set_path", "v": self._policy._set_paths.get(agent.handle, None)
        } for agent in env.agents]
        self.dfs += [{
            "env_time": env._elapsed_steps, "agent_id": agent.handle, "k": "alternatives", "v": self._policy.alternatives.get(agent.handle, None)
        } for agent in env.agents]
        self.dfs += [{
            "env_time": env._elapsed_steps, "agent_id": agent.handle, "k": "num_blocked", "v": self._policy.num_blocked.get(agent.handle, None)
        } for agent in env.agents]
        self.dfs += [{
            "env_time": env._elapsed_steps, "agent_id": agent.handle, "k": "agent_waypoints_done",
            "v": self._policy.agent_waypoints_done.get(agent.handle, None)
        } for agent in env.agents]
        self.dfs += [{
            "env_time": env._elapsed_steps, "agent_id": agent.handle, "k": "agent_waypoints_tried",
            "v": self._policy.agent_waypoints_tried.get(agent.handle, None)
        } for agent in env.agents]
        self.dfs += [{
            "env_time": env._elapsed_steps, "agent_id": agent.handle, "k": "agent_can_move", "v": self._policy.agent_can_move.get(agent.handle, None)
        } for agent in env.agents]
        self.dfs += [{
            "env_time": env._elapsed_steps, "agent_id": agent.handle, "k": "opp_agent_map", "v": self._policy.opp_agent_map.get(agent.handle, None)
        } for agent in env.agents]

    def on_episode_end(
            self,
            *,
            env: Optional[RailEnv] = None,
            data_dir: Path = None,
            **kwargs,
    ) -> None:
        df = pd.DataFrame.from_records(self.dfs)
        df.to_csv(data_dir / "dla_debug.csv")
