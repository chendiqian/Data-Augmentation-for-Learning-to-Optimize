import copy
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def drop_cons(data: HeteroData, drop_idx: np.ndarray) -> Tuple[torch.Tensor, torch.FloatTensor]:
    edge_index = data[('cons', 'to', 'vals')].edge_index.numpy()
    keep_edge_mask = ~np.isin(edge_index[0], drop_idx)

    edge_index = edge_index[:, keep_edge_mask]
    _, remapped_a = np.unique(edge_index[0], return_inverse=True)
    edge_index[0] = remapped_a

    new_edge_index = torch.from_numpy(edge_index).long()
    new_edge_attr = data[('cons', 'to', 'vals')].edge_attr[keep_edge_mask]
    return new_edge_index, new_edge_attr


def count_parameters(model: torch.nn.Module):
    """Source: https://stackoverflow.com/a/62508086"""
    # table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        # table.add_row([name, params])
        total_params += params
    # logger.info(f"\n{str(table)}")
    return total_params
