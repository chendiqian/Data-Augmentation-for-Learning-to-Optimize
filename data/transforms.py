from typing import Tuple, Sequence, List, Any

from random import choices
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, subgraph


def calc_norm(data: Data) -> Data:
    edge_index = data.edge_index
    row, col = edge_index
    deg_src = degree(row, data.num_nodes, dtype=torch.float) + 1.
    deg_src_inv_sqrt = deg_src.pow(-0.5)
    deg_src_inv_sqrt[deg_src_inv_sqrt == float('inf')] = 0
    deg_dst = degree(col, data.num_nodes, dtype=torch.float) + 1.
    deg_dst_inv_sqrt = deg_dst.pow(-0.5)
    deg_dst_inv_sqrt[deg_dst_inv_sqrt == float('inf')] = 0
    norm = deg_src_inv_sqrt[row] * deg_dst_inv_sqrt[col]
    data.norm = norm
    return data


class GCNNorm(BaseTransform):
    # adapted from
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/gcn_norm.html#GCNNorm
    def __init__(self):
        pass

    def forward(self, data):
        if isinstance(data, Sequence):
            return [calc_norm(d) for d in data]
        else:
            return calc_norm(data)


class RandomDropNode(BaseTransform):
    def __init__(self, p):
        assert 0 < p < 1
        self.p = p

    def create_sample(self, data: Data) -> Data:
        n = data.num_nodes
        node_mask = torch.rand(n) > self.p

        # modify cons 2 vals edge_index
        edge_index, edge_attr = subgraph(
            subset=node_mask,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            relabel_nodes=True,
            return_edge_mask=False)

        return Data(x=data.x[node_mask], edge_index=edge_index, edge_attr=edge_attr)

    def forward(self, data: Data) -> Data:
        data = self.create_sample(data)
        return data


class IdentityAugmentation(BaseTransform):
    def __init__(self):
        pass

    def forward(self, data: Any) -> Any:
        return data


class RandomMaskNodeAttr(BaseTransform):
    def __init__(self, p):
        assert 0 < p < 1
        self.p = p

    def forward(self, data: Data) -> Data:
        n = data.num_nodes
        node_masked = torch.rand(n) < self.p

        data.node_masked = node_masked
        return data


class AugmentWrapper(BaseTransform):
    """
    Return 2 views of the graph
    """
    def __init__(self, transforms: List):
        self.transforms = transforms

    def forward(self, data: Data) -> Tuple[Data, Data]:
        while True:
            # I don't want 2 identity
            t1, t2 = choices(self.transforms, k=2)
            if not (isinstance(t1, IdentityAugmentation) and isinstance(t2, IdentityAugmentation)):
                break

        data1 = t1(data)
        data2 = t2(data)
        return data1, data2
