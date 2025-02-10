from typing import Tuple

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, bipartite_subgraph


class GCNNorm(BaseTransform):
    # adapted from
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/gcn_norm.html#GCNNorm
    def __init__(self):
        pass

    def forward(self, data: HeteroData) -> HeteroData:
        for src, rel, dst in data.edge_index_dict.keys():
            edge_index = data[(src, rel, dst)].edge_index
            row, col = edge_index
            deg_src = degree(row, data[src].num_nodes, dtype=torch.float) + 1.
            deg_src_inv_sqrt = deg_src.pow(-0.5)
            deg_src_inv_sqrt[deg_src_inv_sqrt == float('inf')] = 0
            deg_dst = degree(col, data[dst].num_nodes, dtype=torch.float) + 1.
            deg_dst_inv_sqrt = deg_dst.pow(-0.5)
            deg_dst_inv_sqrt[deg_dst_inv_sqrt == float('inf')] = 0
            norm = deg_src_inv_sqrt[row] * deg_dst_inv_sqrt[col]
            data[(src, rel, dst)].norm = norm
            if src != dst:
                data[(dst, rel, src)].norm = norm
        return data


class RandomDropNode(BaseTransform):
    """
    Trivially drop variable and constraint nodes.
    This will violate the original LP problem.
    """
    def __init__(self, p):
        assert 0 < p < 1
        self.p = p

    def create_sample(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        cons_node_mask = torch.rand(m) > self.p
        vals_node_mask = torch.rand(n) > self.p

        # modify cons 2 vals edge_index
        c2v_edge_index, c2v_edge_attr = bipartite_subgraph(
            subset=(cons_node_mask.bool(), vals_node_mask.bool()),
            edge_index=data[('cons', 'to', 'vals')].edge_index,
            edge_attr=data[('cons', 'to', 'vals')].edge_attr,
            relabel_nodes=True,
            size=(m, n),
            return_edge_mask=False)

        # modify obj 2 cons edge_index
        o2c_edge_index, o2c_edge_attr = bipartite_subgraph(
            subset=(torch.ones(1).bool(), cons_node_mask.bool()),
            edge_index=data[('obj', 'to', 'cons')].edge_index,
            edge_attr=data[('obj', 'to', 'cons')].edge_attr,
            relabel_nodes=True,
            size=(1, m),
            return_edge_mask=False)

        # modify obj 2 vals edge_index
        o2v_edge_index, o2v_edge_attr = bipartite_subgraph(
            subset=(torch.ones(1).bool(), vals_node_mask.bool()),
            edge_index=data[('obj', 'to', 'vals')].edge_index,
            edge_attr=data[('obj', 'to', 'vals')].edge_attr,
            relabel_nodes=True,
            size=(1, n),
            return_edge_mask=False)

        new_data = data.__class__(
                cons={
                    'num_nodes': cons_node_mask.sum(),
                    'x': data['cons'].x[cons_node_mask],
                },
                vals={
                    'num_nodes': vals_node_mask.sum(),
                    'x': data['vals'].x[vals_node_mask],
                },
                obj={
                    'num_nodes': 1,
                    'x': data['obj'].x,
                },
                cons__to__vals={'edge_index': c2v_edge_index,
                                'edge_attr': c2v_edge_attr},
                obj__to__vals={'edge_index': o2v_edge_index,
                               'edge_attr': o2v_edge_attr},
                obj__to__cons={'edge_index': o2c_edge_index,
                               'edge_attr': o2c_edge_attr},
                q=data.q[vals_node_mask],
                b=data.b[cons_node_mask],
        )
        return new_data

    def forward(self, data: HeteroData) -> Tuple[HeteroData, HeteroData]:
        data1 = self.create_sample(data)
        data2 = self.create_sample(data)
        return data1, data2
