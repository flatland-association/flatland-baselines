from typing import List, Sequence, Tuple, Union

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.grid.rail_env_grid import RailEnvTransitions, RailEnvTransitionsEnum
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint


def build_rail(rows: Sequence[Sequence[Union[int, RailEnvTransitionsEnum]]]) -> RailGridTransitionMap:
    """Builds a `RailGridTransitionMap` from a rectangular grid of `RailEnvTransitionsEnum` members (use plain `0` for an empty cell)."""
    rail_map = np.array(rows, dtype=np.uint16)
    return RailGridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=RailEnvTransitions(), grid=rail_map)


def target_waypoints(rail: RailGridTransitionMap, position: Tuple[int, int]) -> List[Waypoint]:
    """
    Builds the final waypoint group for a target `position`: one `Waypoint` per direction of arrival
    that is a valid entry point on `rail` - a bare `Waypoint(position, None)` is deprecated now.
    """
    return [Waypoint(position, d) for d in Grid4TransitionsEnum if rail.is_valid_entry_point((position, d))]
