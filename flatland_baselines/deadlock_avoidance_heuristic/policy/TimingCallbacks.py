from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd

from flatland.callbacks.callbacks import FlatlandCallbacks
from flatland.envs.rail_env import RailEnv


class TimingCallbacks(FlatlandCallbacks[RailEnv]):
    def __init__(self):
        self.records: List[dict] = []

    def on_episode_start(
            self,
            *,
            env: Optional[RailEnv] = None,
            data_dir: Path = None,
            **kwargs,
    ) -> None:
        self.records.append({"env_time": env._elapsed_steps, "event": "episode_start", "time": datetime.now().isoformat()})

    def on_episode_step(
            self,
            *,
            env: Optional[RailEnv] = None,
            data_dir: Path = None,
            **kwargs,
    ) -> None:
        self.records.append({"env_time": env._elapsed_steps, "event": "episode_step", "time": datetime.now().isoformat()})

    def on_episode_end(
            self,
            *,
            env: Optional[RailEnv] = None,
            data_dir: Path = None,
            **kwargs,
    ) -> None:
        self.records.append({"env_time": env._elapsed_steps, "event": "episode_end", "time": datetime.now().isoformat()})
        df = pd.DataFrame.from_records(self.records)
        df.to_csv(data_dir / "timing.csv")
