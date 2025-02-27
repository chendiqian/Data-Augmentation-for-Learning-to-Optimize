from typing import Tuple, List
import math

import numpy as np
import torch
from scipy import sparse as sp
from torch_geometric.data import HeteroData
from torch_geometric.utils import bipartite_subgraph
from torch_scatter import scatter_sum
from torch_sparse import SparseTensor


# Todo: add redundant variables

# Todo: change some existing constraints

class DropInactiveConstraint:
    """
    Drop likely inactive constraints
    """

    def __init__(self, p):
        assert 0 < p < 1
        self.p = p

    def neg(self, data: HeteroData, negatives: int) -> Tuple[HeteroData]:
        raise NotImplementedError

    def __call__(self, data: HeteroData) -> HeteroData:
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

        new_data = data.__class__(
            cons={
                'num_nodes': remain_cons.sum(),
                'x': data['cons'].x[remain_cons],
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            cons__to__vals={'edge_index': c2v_edge_index,
                            'edge_attr': c2v_edge_attr},
            q=data.q,
            b=data.b[remain_cons],
            obj_solution=data.obj_solution,
        )
        return new_data


class AddRedundantConstraint:
    """
    Add more constraints PAx <= Pb + eps.
    """

    def __init__(self, p, affinity=3):
        assert 0 < p < 1
        self.p = p
        self.affinity = affinity

    def neg(self, data: HeteroData, negatives: int) -> List[HeteroData]:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        num_new_cons = int(m * self.p)
        edge_index = data[('cons', 'to', 'vals')].edge_index.numpy()
        A = sp.csr_array((data[('cons', 'to', 'vals')].edge_attr.numpy().squeeze(1),
                          (edge_index[0], edge_index[1])), shape=(m, n))

        neg_samples = []
        # todo: parallelize the matmul
        for i in range(negatives):
            eye_mat = sp.diags_array(np.random.randn(m), format='csr')
            rand_mat = sp.random_array((num_new_cons, m), density=self.affinity / m, format='csr')
            rand_mat.data = np.random.randn(*rand_mat.data.shape)
            mat = sp.vstack([eye_mat, rand_mat])

            A_new = (mat @ A).tocoo()
            edge_index = torch.from_numpy(np.vstack([A_new.row, A_new.col])).long()
            edge_attr = torch.from_numpy(A_new.data)[:, None].float()

            new_b = mat @ data.b.numpy()
            bias = np.random.randn(new_b.shape[0])
            bias[:m] = 0
            new_b += bias
            new_b = torch.from_numpy(new_b).float()

            new_data = data.__class__(
                cons={
                    'num_nodes': num_new_cons + m,
                    'x': torch.empty(m + num_new_cons),
                },
                vals={
                    'num_nodes': n,
                    'x': data['vals'].x,
                },
                cons__to__vals={'edge_index': edge_index,
                                'edge_attr': edge_attr},
                q=data.q,
                b=new_b,
                obj_solution=data.obj_solution,
            )
            neg_samples.append(new_data)

        return neg_samples

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        num_new_cons = int(m * self.p)
        eye_mat = sp.eye_array(m, format='csr')
        rand_mat = sp.random_array((num_new_cons, m), density=self.affinity / m, format='csr')
        mat = sp.vstack([eye_mat, rand_mat])

        edge_index = data[('cons', 'to', 'vals')].edge_index.numpy()
        A = sp.csr_array((data[('cons', 'to', 'vals')].edge_attr.numpy().squeeze(1),
                          (edge_index[0], edge_index[1])), shape=(m, n))
        A_new = (mat @ A).tocoo()
        edge_index = torch.from_numpy(np.vstack([A_new.row, A_new.col])).long()
        edge_attr = torch.from_numpy(A_new.data)[:, None].float()

        new_b = mat @ data.b.numpy()
        bias = np.random.randn(new_b.shape[0])
        bias[bias < 0] = 0.
        bias[:m] = 0
        new_b += bias
        new_b = torch.from_numpy(new_b).float()

        new_data = data.__class__(
            cons={
                'num_nodes': num_new_cons + m,
                'x': torch.empty(m + num_new_cons),
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            cons__to__vals={'edge_index': edge_index,
                            'edge_attr': edge_attr},
            q=data.q,
            b=new_b,
            obj_solution=data.obj_solution,
        )
        return new_data


class ScaleObj:
    """
    c <- a * c
    but it's not really obj value preserving, it is solution preserving
    """

    def __init__(self, p):
        assert p > 0.
        self.p = p

    def neg(self, data: HeteroData, negatives: int) -> Tuple[HeteroData]:
        raise NotImplementedError

    def __call__(self, data: HeteroData) -> HeteroData:

        # scales = abs(1 + N(0, 1) * exp(p - 1))

        noise_scale = math.exp(self.p) - 1.
        # 0. -> 0., 1. -> 1.73

        noise = torch.randn(1) * noise_scale
        scale = (noise + 1.).abs()

        new_data = data.__class__(
            cons={
                'num_nodes': data['cons'].num_nodes,
                'x': data['cons'].x,
            },
            vals={
                'num_nodes': data['vals'].num_nodes,
                'x': data['vals'].x,
            },
            cons__to__vals={'edge_index': data[('cons', 'to', 'vals')].edge_index,
                            'edge_attr': data[('cons', 'to', 'vals')].edge_attr},
            q=data.q * scale,
            b=data.b,
            obj_solution=data.obj_solution * scale,
        )
        return new_data


class ScaleConstraint:
    """
    eps * Ax <= eps * b does not change
    eps > 0
    """

    def __init__(self, p):
        assert p > 0
        # we scale all the constraints, but with variable strength
        self.p = p

    def neg(self, data: HeteroData, negatives: int) -> Tuple[HeteroData]:
        raise NotImplementedError

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes

        A = SparseTensor(row=data[('cons', 'to', 'vals')].edge_index[0],
                         col=data[('cons', 'to', 'vals')].edge_index[1],
                         value=data[('cons', 'to', 'vals')].edge_attr.squeeze(1),
                         sparse_sizes=(m, n), is_sorted=True, trust_data=True)

        # scales = abs(1 + N(0, 1) * exp(p - 1))
        noise_scale = math.exp(self.p) - 1.
        # 0. -> 0., 1. -> 1.73

        noise = torch.randn(m) * noise_scale
        scales = (noise + torch.ones_like(noise)).abs()

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
            cons__to__vals={'edge_index': torch.vstack([A.storage.row(), A.storage.col()]),
                            'edge_attr': A.storage.value()[:, None]},
            q=data.q,
            b=new_b,
            obj_solution=data.obj_solution,
        )
        return new_data


class AddOrthogonalConstraint:
    """
    Add constraint ax <= b
    where a.dot(c) = 0, b is large enough. This would not affect the results.
    """

    def __init__(self, p):
        assert 0 < p < 1
        self.p = p

    def neg(self, data: HeteroData, negatives: int) -> Tuple[HeteroData]:
        raise NotImplementedError

    def __call__(self, data: HeteroData) -> HeteroData:
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

        new_data = data.__class__(
            cons={
                'num_nodes': num_new_cons + m,
                'x': torch.empty(m + num_new_cons),
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            cons__to__vals={'edge_index': new_c2v_edge_index,
                            'edge_attr': new_c2v_edge_attr},
            q=data.q,
            b=new_b,
            obj_solution=data.obj_solution,
        )
        return new_data
