# Source: https://github.com/michael-hahn/pytorch-tree-lstm/blob/master/example_usage.py
import numpy as np
import torch

from flatland.envs.observations import Node
from .treelstm.util import calculate_evaluation_orders


def _gather_node_attributes(node: Node):
    features = [list(node)[:-1]]
    for dir, child in node.childs.items():
        if child == -np.inf:
            continue
        features.extend(_gather_node_attributes(child))
    return features


def _gather_adjacency_list(node: Node, n=0):
    adjacency_list = []
    i = 1
    for child in node.childs.values():
        if child == -np.inf:
            continue
        adjacency_list.append([n, n + i])
        sub_ajacency = _gather_adjacency_list(child, n + i)
        adjacency_list.extend(sub_ajacency)
        i += len(sub_ajacency) + 1

    return adjacency_list


def convert_tree_to_tensors(tree, device=torch.device('cpu')):
    features = _gather_node_attributes(tree)
    adjacency_list = _gather_adjacency_list(tree)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))

    return {
        'features': torch.tensor(features, device=device, dtype=torch.float32),
        'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
        'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
        'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
    }
