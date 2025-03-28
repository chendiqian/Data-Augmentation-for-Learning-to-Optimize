import math
from typing import Tuple

import numpy as np
import torch
from scipy import sparse as sp
from torch_geometric.data import HeteroData
from torch_geometric.utils import bipartite_subgraph
from torch_scatter import scatter_sum


class OracleDropInactiveConstraint:
    """
    Drop definitely inactive constraints
    Just for testing, we should not use the ground truth solutions
    """

    def __init__(self, strength=0.1):
        assert 0 < strength < 1
        self.p = strength

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes

        assert hasattr(data, 'inactive_idx')
        inactive_idx = data.inactive_idx.numpy()

        dropped_cons = np.random.choice(inactive_idx, size=min(int(m * self.p), len(inactive_idx)), replace=False)
        remain_cons = ~np.isin(np.arange(m), dropped_cons)
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
                'num_nodes': remain_cons.sum().item(),
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
            x_solution=data.x_solution,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )
        return new_data

    def __repr__(self):
        return 'OracleDropInactiveConstraint'


class DropInactiveConstraint:
    """
    Drop likely inactive constraints
    """

    def __init__(self, strength=0.1):
        assert 0 < strength < 1
        self.p = strength

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes

        assert hasattr(data, 'heur_idx')
        heur_idx = data.heur_idx.numpy()

        dropped_cons = np.random.choice(heur_idx, size=min(int(m * self.p), len(heur_idx)), replace=False)
        remain_cons = ~np.isin(np.arange(m), dropped_cons)
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
                'num_nodes': remain_cons.sum().item(),
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
            x_solution=data.x_solution,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )
        return new_data

    def __repr__(self):
        return 'DropInactiveConstraint'


class AddRedundantConstraint:
    """
    Add more constraints PAx <= Pb + eps.
    """

    def __init__(self, strength=0.2, affinity=3):
        assert 0 < strength < 1
        self.p = strength
        self.affinity = affinity

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        num_new_cons = int(m * self.p)

        idx = np.random.choice(m, (num_new_cons, self.affinity), replace=True)
        weights = np.random.rand(*idx.shape)

        edge_index = data[('cons', 'to', 'vals')].edge_index.numpy()
        edge_attr = data[('cons', 'to', 'vals')].edge_attr.squeeze(1).numpy()

        A = np.zeros((m, n), dtype=np.float32)
        A[edge_index[0], edge_index[1]] = edge_attr

        extra_A = np.einsum('nk,nkf->nf', weights, A[idx])
        where = np.where(extra_A)
        extra_edge_index = torch.from_numpy(np.vstack(where)).long()
        extra_edge_index[0] += m
        extra_edge_attr = torch.from_numpy(extra_A[where[0], where[1]])[:, None].float()

        new_b = (weights * data.b.numpy()[idx]).sum(1)
        bias = np.random.randn(new_b.shape[0])
        bias[bias < 0] = 0.
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
            cons__to__vals={'edge_index': torch.hstack([data[('cons', 'to', 'vals')].edge_index, extra_edge_index]),
                            'edge_attr': torch.vstack([data[('cons', 'to', 'vals')].edge_attr, extra_edge_attr])},
            q=data.q,
            b=torch.hstack([data.b, new_b]),
            obj_solution=data.obj_solution,
            x_solution=data.x_solution,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )
        return new_data

    def __repr__(self):
        return 'AddRedundantConstraint'


class ScaleObj:
    """
    c <- a * c
    but it's not really obj value preserving, it is solution preserving
    """

    def __init__(self, strength=1.):
        assert strength > 0.
        self.p = strength

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
            x_solution=data.x_solution,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )
        return new_data

    def __repr__(self):
        return 'ScaleObj'


class ScaleConstraint:
    """
    eps * Ax <= eps * b does not change
    eps > 0
    """

    def __init__(self, strength=1.):
        assert strength > 0
        # we scale all the constraints, but with variable strength
        self.p = strength

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes

        # scales = abs(1 + N(0, 1) * exp(p - 1))
        noise_scale = math.exp(self.p) - 1.
        # 0. -> 0., 1. -> 1.73

        noise = torch.randn(m) * noise_scale
        scales = (noise + torch.ones_like(noise)).abs()

        new_b = data.b * scales

        edge_index = data[('cons', 'to', 'vals')].edge_index
        new_data = data.__class__(
            cons={
                'num_nodes': m,
                'x': data['cons'].x,
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            cons__to__vals={'edge_index': edge_index,
                            'edge_attr': data[('cons', 'to', 'vals')].edge_attr * scales[edge_index[0], None]},
            q=data.q,
            b=new_b,
            obj_solution=data.obj_solution,
            x_solution=data.x_solution,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )
        return new_data

    def __repr__(self):
        return 'ScaleConstraint'


class ScaleCoordinate:
    """
    Imagine the coordinates are expanded, with c also expanded correspondingly,
    then the optimal solution is scaled, but the objective is unchanged
    """

    def __init__(self, strength=1.):
        assert strength > 0
        # we scale all the constraints, but with variable strength
        self.p = strength

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes

        # scales = abs(1 + N(0, 1) * exp(p - 1))
        noise_scale = math.exp(self.p) - 1.
        # 0. -> 0., 1. -> 1.73

        noise = torch.randn(n) * noise_scale
        scales = (noise + torch.ones_like(noise)).abs()

        cols = data[('cons', 'to', 'vals')].edge_index[1]

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
                            'edge_attr': data[('cons', 'to', 'vals')].edge_attr * scales[cols, None]},
            q=data.q * scales,
            b=data.b,
            obj_solution=data.obj_solution,
            x_solution=data.x_solution,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )
        return new_data

    def __repr__(self):
        return 'ScaleCoordinate'


class AddSubOrthogonalConstraint:
    """
    Add constraint ax <= b
    where a.dot(c) >= 0, b is large enough. This would not affect the results.
    """

    def __init__(self, strength=0.1):
        assert 0 < strength < 1
        self.p = strength

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes

        def batch_sparse_orthogonal(c, num_new, density=0.1):
            sparsity = int(n * density)

            row = torch.arange(num_new).repeat_interleave(sparsity)
            # m * nnz
            col = np.vstack([np.sort(np.random.choice(n, sparsity, replace=False)) for _ in range(num_new)])
            col = torch.from_numpy(col).long()
            free_values = torch.randn(num_new, sparsity - 1)
            # so that each A @ c = rand > 0
            last_values = ((torch.rand(num_new) - (
                        c[col[:, :-1].reshape(-1)].reshape(num_new, sparsity - 1) * free_values).sum(1))
                           / c[col[:, -1]])
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
        assert data.q.min() >= 0.
        extra_b = extra_data * (torch.where(extra_data > 0, 3, 0))
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
            x_solution=data.x_solution,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )
        return new_data

    def __repr__(self):
        return 'AddSubOrthogonalConstraint'


class AddDumbVariables:
    """
    Add variables with non-negative c value
    """

    def __init__(self, strength=0.1):
        assert 0 < strength < 1
        self.p = strength

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes

        num_new_vars = int(n * self.p)
        density = data[('cons', 'to', 'vals')].edge_index.shape[1] / (m * n)

        selected_idx = np.sort(np.random.choice(num_new_vars * m, int(num_new_vars * m * density), replace=False))
        new_row = selected_idx // num_new_vars
        new_col = selected_idx % num_new_vars + n

        extra_edge_index = torch.from_numpy(np.vstack([new_row, new_col])).long()

        # Todo: maybe not rand, but -1 0 +1 for some class of LP
        # much be positive, otherwise would relax the constraints and might change the solution
        extra_edge_attr = torch.rand(extra_edge_index.shape[1], 1)

        new_data = data.__class__(
            cons={
                'num_nodes': m,
                'x': data['cons'].x,
            },
            vals={
                'num_nodes': n + num_new_vars,
                'x': torch.empty(n + num_new_vars),
            },
            cons__to__vals={'edge_index': torch.hstack([data[('cons', 'to', 'vals')].edge_index, extra_edge_index]),
                            'edge_attr': torch.vstack([data[('cons', 'to', 'vals')].edge_attr, extra_edge_attr])},
            # added c must be non-negative, otherwise might change the solution
            q=torch.cat([data.q, torch.rand(num_new_vars)]),
            b=data.b,
            obj_solution=data.obj_solution,
            x_solution=data.x_solution,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )
        return new_data

    def __repr__(self):
        return 'AddDumbVariables'


class CombinedDualViewAugmentations:
    def __init__(self,
                 dropinactiveconstraint: float = 0.,
                 addredundantconstraint: float = 0.4,
                 scaleconstraint: float = 1.,
                 scalecoordinate: float = 1.,
                 adddumbvariables: float = 0.4, **kwargs):
        self.aug_list = {'dropinactiveconstraint': dropinactiveconstraint,
                         'addredundantconstraint': addredundantconstraint,
                         'scaleconstraint': scaleconstraint,
                         'scalecoordinate': scalecoordinate,
                         'adddumbvariables': adddumbvariables}

    def forward(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        edge_index = data[('cons', 'to', 'vals')].edge_index.numpy()

        A = sp.csr_array((data[('cons', 'to', 'vals')].edge_attr.numpy().squeeze(1),
                          (edge_index[0], edge_index[1])), shape=(m, n))
        b = data.b.numpy()
        c = data.q.numpy()

        # drop inactive constraints
        if self.aug_list['dropinactiveconstraint'] > 0:
            p = self.aug_list['dropinactiveconstraint']
            heur_idx = data.heur_idx.numpy()
            dropped_cons = np.random.choice(heur_idx, size=min(int(m * p), len(heur_idx)), replace=False)
            remain_cons = ~np.isin(np.arange(m), dropped_cons)
            A = A[remain_cons, :]
            b = b[remain_cons]

            m = remain_cons.sum()

        # AddRedundantConstraint
        affinity = 3
        num_new_cons = int(m * self.aug_list['addredundantconstraint'])
        rows = np.hstack([np.arange(m), np.arange(num_new_cons).repeat(affinity) + m])
        cols = np.hstack([np.arange(m), np.random.randint(low=0, high=m, size=affinity * num_new_cons)])
        values = np.hstack([np.ones(m) * 1., np.random.rand(affinity * num_new_cons)])
        rand_mat = sp.csr_array((values, (rows, cols)), shape=(num_new_cons + m, m))

        b_bias = np.random.randn(m + num_new_cons)
        b_bias[b_bias < 0] = 0.
        b_bias[:m] = 0
        m = num_new_cons + m

        # scale constraints
        noise_scale = math.exp(self.aug_list['scaleconstraint']) - 1.
        noise = np.random.randn(m) * noise_scale
        scales = np.abs(noise + np.ones_like(noise))
        rand_mat = rand_mat * scales[:, None]

        b = rand_mat @ b + b_bias
        A = rand_mat @ A

        # scale coordinates
        noise_scale = math.exp(self.aug_list['scalecoordinate']) - 1.

        noise = np.random.randn(n) * noise_scale
        scales = np.abs(noise + np.ones_like(noise))
        A = A * scales[None]
        c = c * scales

        # AddSubOrthogonalConstraint
        A = A.tocoo()

        num_new_vars = int(n * self.aug_list['adddumbvariables'])
        density = A.nnz / np.prod(A.shape)

        selected_idx = np.sort(np.random.choice(num_new_vars * m, int(num_new_vars * m * density), replace=False))
        new_row = selected_idx // num_new_vars
        new_col = selected_idx % num_new_vars + n

        edge_index = np.vstack([A.row, A.col])
        extra_edge_index = np.vstack([new_row, new_col])
        edge_index = np.hstack([edge_index, extra_edge_index])
        edge_attr = np.hstack([A.data, np.random.rand(extra_edge_index.shape[1])])
        c = np.hstack([c, np.random.rand(num_new_vars)])

        n = n + num_new_vars

        new_data = data.__class__(
            cons={
                'num_nodes': m,
                'x': torch.empty(m),
            },
            vals={
                'num_nodes': n,
                'x': torch.empty(n),
            },
            cons__to__vals={'edge_index': torch.from_numpy(edge_index).long(),
                            'edge_attr': torch.from_numpy(edge_attr).float()[:, None]},
            # added c must be non-negative, otherwise might change the solution
            q=torch.from_numpy(c).float(),
            b=torch.from_numpy(b).float(),
        )
        return new_data

    def __call__(self, data: HeteroData) -> Tuple[HeteroData, HeteroData]:
        return self.forward(data), self.forward(data)
