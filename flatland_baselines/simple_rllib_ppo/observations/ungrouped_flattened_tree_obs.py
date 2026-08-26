from typing import Optional

import gymnasium as gym
import numpy as np

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.observations import TreeObsForRailEnv, Node
from flatland.envs.rail_env import RailEnv
from flatland.ml.observations.gym_observation_builder import GymObservationBuilder


# TODO backport to flatland-rl
class UngroupedFlattenedTreeObsForRailEnv(GymObservationBuilder[RailEnv, np.ndarray], TreeObsForRailEnv):
    """
    Gym-ified and flattened normalized tree observation without feature grouping.
    """

    NUM_FEATURES = 2
    NUM_BRANCHES = 4

    DEFAULT_VALUE = np.inf
    MIN_VALULE = 0
    MAX_VALULE = 1000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_len_data(tree_depth: int, num_features):
        k = num_features
        for _ in range(tree_depth):
            k = k * UngroupedFlattenedTreeObsForRailEnv.NUM_BRANCHES + num_features
        return k

    def _traverse_subtree(self, node, current_tree_depth: int, max_tree_depth: int) -> np.ndarray:
        if node == -np.inf:
            remaining_depth = max_tree_depth - current_tree_depth
            # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
            num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
            return [-np.inf] * num_remaining_nodes * self.NUM_FEATURES

        data = list(self._node_extractor(node))

        if not node.childs:
            return data

        for direction in TreeObsForRailEnv.tree_explored_actions_char:
            sub_data = self._traverse_subtree(node.childs[direction], current_tree_depth + 1, max_tree_depth)
            data = np.concatenate((data, sub_data))

        return data

    def traverse_tree(self, root_node: Node, max_tree_depth: int) -> np.ndarray:
        """
        This function extracts:

        - `data` (12 features per node):
            - `node.dist_own_target_encountered`
            - `node.dist_other_target_encountered`
            - `node.dist_other_agent_encountered`
            - `node.dist_potential_conflict`
            - `node.dist_unusable_switch`
            - `node.dist_to_next_branch`
            - `node.dist_min_to_target`
            - `node.num_agents_same_direction`
            - `node.num_agents_opposite_direction`
            - `node.num_agents_malfunctioning`
            - `node.speed_min_fractional`
            - `node.num_agents_ready_to_depart`

        Subtrees are traversed depth-first in pre-order (i.e. Node 'N' itself, 'L', 'F', 'R', 'B').
        See `get_len_flattened()` for the length of the flattened structure.
        """
        data = list(self._node_extractor(root_node))

        for direction in UngroupedFlattenedTreeObsForRailEnv.tree_explored_actions_char:
            sub_data = self._traverse_subtree(root_node.childs[direction], 1, max_tree_depth)
            data = np.concatenate((data, sub_data))

        # convert Fraction to float
        data = data.astype(np.float64)

        data[data == -np.inf] = np.inf
        return np.clip(data, self.MIN_VALULE, self.MAX_VALULE, )

    def _node_extractor(self, root_node: Node) -> tuple:
        return root_node[5:7]

    def get(self, handle: Optional[AgentHandle] = 0) -> np.ndarray:
        root_node = super(UngroupedFlattenedTreeObsForRailEnv, self).get(handle)
        self.make_complete_nary(root_node, 0, self.max_depth, UngroupedFlattenedTreeObsForRailEnv.tree_explored_actions_char)
        return self.traverse_tree(root_node, self.max_depth)

    def get_observation_space(self, handle: int = 0) -> gym.Space:
        k = self.get_len_flattened()
        return gym.spaces.Box(low=self.MIN_VALULE, high=self.MAX_VALULE, shape=(k,), dtype=np.float64)

    def get_len_flattened(self):
        """
        The total size `S[k]` of the flattened structure for `max_tree_depth=k` is recursively defined by:
        - `S[0] = NUM_FEATURES`
        - `S[k+1] = S[k] * NUM_BRANCHES + NUM_FEATURES`
        for
        - `NUM_FEATURES=12`
        - `NUM_BRANCHES=4`

        I.e.
        - max_depth=1 -> 60
        - max_depth=2 -> 252
        - max_depth=3 -> 1020
        - ...

        Returns
        -------
        Length of the flattened tree obs.
        """

        k = UngroupedFlattenedTreeObsForRailEnv.NUM_FEATURES
        for _ in range(self.max_depth):
            k = k * UngroupedFlattenedTreeObsForRailEnv.NUM_BRANCHES + UngroupedFlattenedTreeObsForRailEnv.NUM_FEATURES
        return k

    def make_complete_nary(self, node: Node, depth, max_depth, tree_explored_actions_char):
        if depth < max_depth:
            for a in tree_explored_actions_char:
                if a not in node.childs or node.childs[a] == -np.inf:
                    node.childs[a] = Node(*[-np.inf] * 12, {})
                self.make_complete_nary(node.childs[a], depth + 1, max_depth, tree_explored_actions_char)

# def unflatten(flat, index, depth, max_depth, num_features, tree_explored_actions_char):
#     if (flat[index * num_features:index * num_features + num_features] == -np.inf).all():
#         # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
#         sub_tree_size = int((4 ** (max_depth - depth + 1) - 1) / (4 - 1))
#         return -np.inf, index + sub_tree_size
#     n = Node(*flat[index * num_features:index * num_features + num_features], childs={})
#     index += 1
#     if depth < max_depth:
#         for a in tree_explored_actions_char:
#             child, index = unflatten(flat, index, depth + 1, max_depth, num_features, tree_explored_actions_char)
#             n.childs[a] = child
#     return n, index
