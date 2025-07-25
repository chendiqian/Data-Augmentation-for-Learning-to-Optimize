import torch
from torch_cluster import random_walk
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph, bipartite_subgraph, to_undirected, degree

from utils.evaluation import is_qp


class RWSubgraph:
    """
    RW subgraph, for GCC baseline
    """
    def __init__(self, walk_length=100):
        self.walk_length = walk_length

    def __call__(self, data: HeteroData) -> HeteroData:
        _is_qp = is_qp(data)
        m, n = data['cons'].num_nodes, data['vals'].num_nodes

        homo_edge_index = data['cons', 'to', 'vals'].edge_index.clone()  # CANNOT IN-SPACE!!!!
        homo_edge_index[1] += m
        homo_edge_index = to_undirected(homo_edge_index, num_nodes=m + n)
        if _is_qp:
            homo_edge_index = torch.hstack([homo_edge_index, data['vals', 'to', 'vals'].edge_index + m])
        deg = degree(homo_edge_index[0], m + n)

        # GCC starts with the one with max degree
        # https://github.com/THUDM/GCC/blob/master/gcc/datasets/graph_dataset.py#L369
        walk = random_walk(homo_edge_index[0],
                           homo_edge_index[1],
                           deg.argmax().unsqueeze(0),
                           walk_length=self.walk_length,
                           num_nodes=m + n)

        nodes = walk.unique()
        cons = nodes[nodes < m]
        vals = nodes[nodes >= m] - m

        # modify cons 2 vals edge_index
        c2v_edge_index, c2v_edge_attr = bipartite_subgraph(
            subset=(cons, vals),
            edge_index=data[('cons', 'to', 'vals')].edge_index,
            edge_attr=data[('cons', 'to', 'vals')].edge_attr,
            relabel_nodes=True,
            size=(m, n),
            return_edge_mask=False)

        new_data = data.__class__(
            cons={
                'num_nodes': len(cons),
                'x': data['cons'].x[cons],
            },
            vals={
                'num_nodes': len(vals),
                'x': data['vals'].x[vals],
            },
            cons__to__vals={'edge_index': c2v_edge_index,
                            'edge_attr': c2v_edge_attr},
            q=data.q[vals],
            b=data.b[cons],
            obj_solution=data.obj_solution,  # this is actually not correct
        )

        if _is_qp:
            v2v_edge_index, v2v_edge_attr = subgraph(subset=vals,
                                                     edge_index=data[('vals', 'to', 'vals')].edge_index,
                                                     edge_attr=data[('vals', 'to', 'vals')].edge_attr,
                                                     relabel_nodes=True,
                                                     num_nodes=n)
            new_data[('vals', 'to', 'vals')].edge_index = v2v_edge_index
            new_data[('vals', 'to', 'vals')].edge_attr = v2v_edge_attr
        return new_data
