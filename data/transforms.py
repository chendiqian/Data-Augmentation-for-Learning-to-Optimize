from typing import Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, subgraph


class GCNNorm(BaseTransform):
    # adapted from
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/gcn_norm.html#GCNNorm
    def __init__(self):
        pass

    def forward(self, data: Data) -> Data:
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


class RandomDropNode(BaseTransform):
    """
    Trivially drop variable and constraint nodes.
    This will violate the original LP problem.
    """
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

    def forward(self, data: Data) -> Tuple[Data, Data]:
        data1 = self.create_sample(data)
        data2 = self.create_sample(data)
        return data1, data2
