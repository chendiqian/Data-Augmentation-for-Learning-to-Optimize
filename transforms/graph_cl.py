import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import bipartite_subgraph, add_random_edge


class GraphCLDropNode:
    """
    Trivially drop variable and constraint nodes.
    This will violate the original LP problem.
    """

    def __init__(self, strength):
        assert 0 < strength < 1
        self.p = strength

    def __call__(self, data: HeteroData) -> HeteroData:
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

        new_data = data.__class__(
            cons={
                'num_nodes': cons_node_mask.sum(),
                'x': data['cons'].x[cons_node_mask],
            },
            vals={
                'num_nodes': vals_node_mask.sum(),
                'x': data['vals'].x[vals_node_mask],
            },
            cons__to__vals={'edge_index': c2v_edge_index,
                            'edge_attr': c2v_edge_attr},
            q=data.q[vals_node_mask],
            b=data.b[cons_node_mask],
            obj_solution=data.obj_solution,  # this is actually not correct
        )
        return new_data


class GraphCLPerturbEdge:
    """
    Trivially drop or add A connections.
    This will violate the original LP problem.
    """

    def __init__(self, strength):
        assert 0 < strength < 1
        self.p = strength

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        edge_index = data[('cons', 'to', 'vals')].edge_index
        ne = edge_index.shape[1]

        # remove edges
        edge_mask = torch.rand(ne) > self.p
        c2v_edge_index = data[('cons', 'to', 'vals')].edge_index[:, edge_mask]
        c2v_edge_attr = data[('cons', 'to', 'vals')].edge_attr[edge_mask, :]

        # add edges
        added_edge_index = add_random_edge(edge_index, p=self.p, num_nodes=(m, n))[1]
        added_edge_attr = torch.randn(added_edge_index.shape[1], 1)

        new_data = data.__class__(
            cons={
                'num_nodes': m,
                'x': data['cons'].x,
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            cons__to__vals={'edge_index': torch.hstack([c2v_edge_index, added_edge_index]),
                            'edge_attr': torch.vstack([c2v_edge_attr, added_edge_attr])},
            q=data.q,
            b=data.b,
            obj_solution=data.obj_solution,  # this is actually not correct
        )
        return new_data


class GraphCLMaskNode:
    """
    Trivially mask variable and constraint nodes.
    This will violate the original LP problem.
    """

    def __init__(self, strength):
        assert 0 < strength < 1
        self.p = strength

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        cons_node_mask = torch.rand(m) < self.p
        vals_node_mask = torch.rand(n) < self.p

        new_data = data.__class__(
            cons={
                'num_nodes': m,
                'x': data['cons'].x,
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            cons__to__vals={'edge_index': data[('cons', 'to', 'vals')].edge_index,
                            'edge_attr': data[('cons', 'to', 'vals')].edge_attr},
            q=data.q.masked_fill(vals_node_mask, 0),
            b=data.b.masked_fill(cons_node_mask, 0),
            obj_solution=data.obj_solution,  # this is actually not correct
        )
        return new_data
