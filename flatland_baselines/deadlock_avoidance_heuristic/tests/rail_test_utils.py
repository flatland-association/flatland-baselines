from typing import Sequence, Union

import numpy as np

from flatland.envs.grid.rail_env_grid import RailEnvTransitions, RailEnvTransitionsEnum
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap


def build_rail(rows: Sequence[Sequence[Union[int, RailEnvTransitionsEnum]]]) -> RailGridTransitionMap:
    """Builds a `RailGridTransitionMap` from a rectangular grid of `RailEnvTransitionsEnum` members (use plain `0` for an empty cell)."""
    rail_map = np.array(rows, dtype=np.uint16)
    return RailGridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=RailEnvTransitions(), grid=rail_map)
