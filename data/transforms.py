from typing import Tuple, List
from random import choices, choice

import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor
from torch_scatter import scatter_sum
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

    def forward(self, data: HeteroData) -> HeteroData:
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


class DropInactiveConstraint(BaseTransform):
    """
    Drop likely inactive constraints
    """

    def __init__(self, p):
        assert 0 < p < 1
        self.p = p

    def forward(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        active_sort_idx = data.active_sort_idx.numpy()
        z = np.arange(m) - m // 2
        prob = 1 / (1 + np.exp(-z))
        prob /= prob.sum()
        dropped_cons = np.random.choice(active_sort_idx, size=int(m * self.p), replace=False, p=prob)
        remain_cons = ~np.in1d(np.arange(m), dropped_cons)
        remain_cons = torch.from_numpy(remain_cons)

        # modify cons 2 vals edge_index
        c2v_edge_index, c2v_edge_attr = bipartite_subgraph(
            subset=(remain_cons, torch.ones(n, dtype=torch.bool)),
            edge_index=data[('cons', 'to', 'vals')].edge_index,
            edge_attr=data[('cons', 'to', 'vals')].edge_attr,
            relabel_nodes=True,
            size=(m, n),
            return_edge_mask=False)

        # modify obj 2 cons edge_index
        o2c_edge_index, o2c_edge_attr = bipartite_subgraph(
            subset=(torch.ones(1).bool(), remain_cons),
            edge_index=data[('obj', 'to', 'cons')].edge_index,
            edge_attr=data[('obj', 'to', 'cons')].edge_attr,
            relabel_nodes=True,
            size=(1, m),
            return_edge_mask=False)

        new_data = data.__class__(
            cons={
                'num_nodes': remain_cons.sum(),
                'x': data['cons'].x[remain_cons],
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            obj={
                'num_nodes': 1,
                'x': data['obj'].x,
            },
            cons__to__vals={'edge_index': c2v_edge_index,
                            'edge_attr': c2v_edge_attr},
            obj__to__vals={'edge_index': data[('obj', 'to', 'vals')].edge_index,
                           'edge_attr': data[('obj', 'to', 'vals')].edge_attr},
            obj__to__cons={'edge_index': o2c_edge_index,
                           'edge_attr': o2c_edge_attr},
            q=data.q,
            b=data.b[remain_cons],
        )
        return new_data


class AddRedundantConstraint(BaseTransform):
    """
    Add more constraints PAx <= Pb + eps.
    """

    def __init__(self, p, affinity=3):
        assert 0 < p < 1
        self.p = p
        self.affinity = affinity

    def forward(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        num_new_cons = int(m * self.p)
        rand_mat = sp.random_array((num_new_cons, m), density=self.affinity / m, format='csr')
        edge_index = data[('cons', 'to', 'vals')].edge_index.numpy()
        A = sp.csr_array((data[('cons', 'to', 'vals')].edge_attr.numpy().squeeze(1),
                          (edge_index[0], edge_index[1])), shape=(m, n))
        A_new = (rand_mat @ A).tocoo()
        new_edge_index = np.vstack([A_new.row, A_new.col])
        new_edge_attr = A_new.data
        new_edge_index[0] += m

        extra_b = rand_mat @ data.b.numpy()
        extra_b += np.random.rand(extra_b.shape[0])
        new_b = torch.cat([data.b, torch.from_numpy(extra_b).float()], dim=0)

        c2v_edge_index = torch.cat([data[('cons', 'to', 'vals')].edge_index,
                                    torch.from_numpy(new_edge_index).long()], dim=1)
        c2v_edge_attr = torch.cat([data[('cons', 'to', 'vals')].edge_attr,
                                   torch.from_numpy(new_edge_attr[:, None]).float()], dim=0)

        o2c_edge_index = torch.vstack([torch.zeros(m + num_new_cons).long(),
                                       torch.arange(m + num_new_cons)])
        o2c_edge_attr = new_b[:, None]

        new_data = data.__class__(
            cons={
                'num_nodes': num_new_cons + m,
                'x': torch.empty(m + num_new_cons),
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            obj={
                'num_nodes': 1,
                'x': data['obj'].x,
            },
            cons__to__vals={'edge_index': c2v_edge_index,
                            'edge_attr': c2v_edge_attr},
            obj__to__vals={'edge_index': data[('obj', 'to', 'vals')].edge_index,
                           'edge_attr': data[('obj', 'to', 'vals')].edge_attr},
            obj__to__cons={'edge_index': o2c_edge_index,
                           'edge_attr': o2c_edge_attr},
            q=data.q,
            b=new_b,
        )
        return new_data


class ScaleInstance(BaseTransform):
    """
    eps * Ax <= eps * b does not change
    eps > 0
    """

    def __init__(self, p):
        assert 0 < p < 1
        self.p = p

    def forward(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes

        A = SparseTensor(row=data[('cons', 'to', 'vals')].edge_index[0],
                         col=data[('cons', 'to', 'vals')].edge_index[1],
                         value=data[('cons', 'to', 'vals')].edge_attr.squeeze(1),
                         sparse_sizes=(m, n), is_sorted=True, trust_data=True)
        scales = torch.abs(torch.randn(m))
        scales[torch.rand(m) > self.p] = 1.
        A = A * scales[:, None]
        new_b = data.b * scales

        new_data = data.__class__(
            cons={
                'num_nodes': m,
                'x': data['cons'].x,
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            obj={
                'num_nodes': 1,
                'x': data['obj'].x,
            },
            cons__to__vals={'edge_index': torch.vstack([A.storage.row(), A.storage.col()]),
                            'edge_attr': A.storage.value()[:, None]},
            obj__to__vals={'edge_index': data[('obj', 'to', 'vals')].edge_index,
                           'edge_attr': data[('obj', 'to', 'vals')].edge_attr},
            obj__to__cons={'edge_index': data[('obj', 'to', 'cons')].edge_index,
                           'edge_attr': new_b[:, None]},
            q=data.q,
            b=new_b,
        )
        return new_data


class AddOrthogonalConstraint(BaseTransform):
    """
    Add constraint ax <= b
    where a.dot(c) = 0, b is large enough. This would not affect the results.
    """

    def __init__(self, p):
        assert 0 < p < 1
        self.p = p

    def forward(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes

        def batch_sparse_orthogonal(c, num_new, density=0.1):
            sparsity = int(n * density)

            row = torch.arange(num_new).repeat_interleave(sparsity)
            # m * nnz
            col = np.vstack([np.sort(np.random.choice(n, sparsity, replace=False)) for _ in range(num_new)])
            col = torch.from_numpy(col).long()
            free_values = torch.randn(num_new, sparsity - 1)
            last_values = -(c[col[:, :-1].reshape(-1)].reshape(num_new, sparsity - 1) * free_values).sum(1) / c[col[:, -1]]
            values = torch.cat([free_values, last_values[:, None]], dim=1)
            values /= values.max(dim=1, keepdim=True).values
            return row, col.reshape(-1), values.reshape(-1)

        num_new_cons = int(m * self.p)
        extra_row, extra_col, extra_data = batch_sparse_orthogonal(
            data.q,
            num_new_cons,
            data[('cons', 'to', 'vals')].edge_index.shape[1] / (m * n))

        # a heuristic, we need a large enough b so that not to violate current feasible region
        # ideally we should narrow the bounds of all the variables and get an upper bound of b, but that's hard
        # if c_i is large, the solution x_i is probably small
        assert data.q.min() >= 0.
        extra_b = extra_data * (
            torch.where(extra_data > 0,
                        torch.clamp(1. / (data.q + 1.e-7), max=5.)[extra_col],
                        0)
        )
        extra_b = scatter_sum(extra_b, extra_row, dim=0)
        new_b = torch.cat([data.b, extra_b], dim=0)
        extra_edge_index = torch.vstack([extra_row + m, extra_col])
        new_c2v_edge_index = torch.cat([data[('cons', 'to', 'vals')].edge_index, extra_edge_index], dim=1)
        new_c2v_edge_attr = torch.cat([data[('cons', 'to', 'vals')].edge_attr, extra_data[:, None]], dim=0)

        o2c_edge_index = torch.vstack([torch.zeros(m + num_new_cons).long(),
                                       torch.arange(m + num_new_cons)])
        o2c_edge_attr = new_b[:, None]
        new_data = data.__class__(
            cons={
                'num_nodes': num_new_cons + m,
                'x': torch.empty(m + num_new_cons),
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            obj={
                'num_nodes': 1,
                'x': data['obj'].x,
            },
            cons__to__vals={'edge_index': new_c2v_edge_index,
                            'edge_attr': new_c2v_edge_attr},
            obj__to__vals={'edge_index': data[('obj', 'to', 'vals')].edge_index,
                           'edge_attr': data[('obj', 'to', 'vals')].edge_attr},
            obj__to__cons={'edge_index': o2c_edge_index,
                           'edge_attr': o2c_edge_attr},
            q=data.q,
            b=new_b,
        )
        return new_data


class AugmentWrapper(BaseTransform):
    """
    Return 2 views of the graph
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def forward(self, data: HeteroData) -> Tuple[HeteroData, HeteroData]:
        # while True:
        #     # I don't want 2 identity
        #     t1, t2 = choices(self.transforms, k=2)
        #     if not (isinstance(t1, IdentityAugmentation) and isinstance(t2, IdentityAugmentation)):
        #         break
        t1, t2 = choices(self.transforms, k=2)

        data1 = t1(data)
        data2 = t2(data)
        return data1, data2


# Todo: add redundant variables

# Todo: change some existing constraints


TRANSFORM_CODEBOOK = {
    '0': RandomDropNode,
    '1': DropInactiveConstraint,
    '2': AddRedundantConstraint,
    '3': ScaleInstance,
    '4': AddOrthogonalConstraint,
}
