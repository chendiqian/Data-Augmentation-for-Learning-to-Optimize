import math
from typing import Dict
import random

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_scatter import scatter_sum

from utils.evaluation import is_qp, oracle_inactive_constraints, inactive_contraint_heuristic
from utils.models import drop_cons


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

        if hasattr(data, 'inactive_idx'):
            inactive_idx = data.inactive_idx.numpy()
        else:
            inactive_idx = oracle_inactive_constraints(data)

        drop_idx = np.random.choice(inactive_idx, size=min(int(m * self.p), len(inactive_idx)), replace=False)
        c2v_edge_index, c2v_edge_attr = drop_cons(data, drop_idx)
        remain_cons = ~np.isin(np.arange(m), drop_idx)

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

        if is_qp(data):
            new_data[('vals', 'to', 'vals')].edge_index = data[('vals', 'to', 'vals')].edge_index
            new_data[('vals', 'to', 'vals')].edge_attr = data[('vals', 'to', 'vals')].edge_attr
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
        if is_qp(data):
            raise NotImplementedError

        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        assert hasattr(data, 'heur_idx')

        if hasattr(data, 'heur_idx'):
            heur_idx = data.heur_idx.numpy()
        else:
            heur_idx = inactive_contraint_heuristic(data)

        drop_idx = np.random.choice(heur_idx, size=min(int(m * self.p), len(heur_idx)), replace=False)
        new_edge_index, new_edge_attr = drop_cons(data, drop_idx)
        remain_cons = ~np.isin(np.arange(m), drop_idx)

        new_data = data.__class__(
            cons={
                'num_nodes': remain_cons.sum().item(),
                'x': data['cons'].x[remain_cons],
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            cons__to__vals={'edge_index': new_edge_index,
                            'edge_attr': new_edge_attr},
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

        if is_qp(data):
            new_data[('vals', 'to', 'vals')].edge_index = data[('vals', 'to', 'vals')].edge_index
            new_data[('vals', 'to', 'vals')].edge_attr = data[('vals', 'to', 'vals')].edge_attr
        return new_data

    def __repr__(self):
        return 'AddRedundantConstraint'


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

        if is_qp(data):
            new_data[('vals', 'to', 'vals')].edge_index = data[('vals', 'to', 'vals')].edge_index
            new_data[('vals', 'to', 'vals')].edge_attr = data[('vals', 'to', 'vals')].edge_attr
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
            x_solution=data.x_solution / scales,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )

        if is_qp(data):
            Qrows, Qcols = data[('vals', 'to', 'vals')].edge_index
            new_data[('vals', 'to', 'vals')].edge_index = data[('vals', 'to', 'vals')].edge_index
            new_data[('vals', 'to', 'vals')].edge_attr = (data[('vals', 'to', 'vals')].edge_attr *
                                                          scales[Qcols, None] * scales[Qrows, None])
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
        if is_qp(data):
            raise NotImplementedError

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

        if is_qp(data):
            row = torch.arange(n)
            col = torch.full((n,), n, dtype=torch.long)
            extra_v2v_edge_index = torch.hstack([torch.vstack([row, col]),
                                                 torch.vstack([col, row]),
                                                 torch.tensor([[n], [n]], dtype=torch.long)])
            extra_v2v_edge_attr = torch.rand(n)
            Q_dens = data[('vals', 'to', 'vals')].edge_index.shape[1] / (n * n)
            extra_v2v_edge_attr[np.random.rand(n) > Q_dens] = 0.
            extra_v2v_edge_attr = torch.hstack([extra_v2v_edge_attr, extra_v2v_edge_attr, torch.rand(1)])

            new_data[('vals', 'to', 'vals')].edge_index = torch.hstack([data[('vals', 'to', 'vals')].edge_index,
                                                                        extra_v2v_edge_index])
            new_data[('vals', 'to', 'vals')].edge_attr = torch.vstack([data[('vals', 'to', 'vals')].edge_attr,
                                                                       extra_v2v_edge_attr[:, None]])
        return new_data

    def __repr__(self):
        return 'AddDumbVariables'


class ComboPreservedTransforms:
    def __init__(self, tf_dict: Dict):
        strengths = tf_dict.values()
        assert max(strengths) > 0, "At least 1 transformation!"

        assert not ('OracleDropInactiveConstraint' in tf_dict and
                    'DropInactiveConstraint' in tf_dict and
                    tf_dict['OracleDropInactiveConstraint'] * tf_dict['DropInactiveConstraint'] > 0), "Cannot both"

        if 'OracleDropInactiveConstraint' in tf_dict and tf_dict['OracleDropInactiveConstraint'] > 0:
            self.oracle_drop_c = OracleDropInactiveConstraint(tf_dict['OracleDropInactiveConstraint'])
        else:
            self.oracle_drop_c = None

        if 'DropInactiveConstraint' in tf_dict and tf_dict['DropInactiveConstraint'] > 0:
            self.drop_c = DropInactiveConstraint(tf_dict['DropInactiveConstraint'])
        else:
            self.drop_c = None

        if 'AddRedundantConstraint' in tf_dict and tf_dict['AddRedundantConstraint'] > 0:
            self.add_c = AddRedundantConstraint(tf_dict['AddRedundantConstraint'])
        else:
            self.add_c = None

        if 'ScaleConstraint' in tf_dict and tf_dict['ScaleConstraint'] > 0:
            self.scale_c = ScaleConstraint(tf_dict['ScaleConstraint'])
        else:
            self.scale_c = None

        if 'ScaleCoordinate' in tf_dict and tf_dict['ScaleCoordinate'] > 0:
            self.scale_v = ScaleCoordinate(tf_dict['ScaleCoordinate'])
        else:
            self.scale_v = None

        if 'AddDumbVariables' in tf_dict and tf_dict['AddDumbVariables'] > 0:
            self.add_v = AddDumbVariables(tf_dict['AddDumbVariables'])
        else:
            self.add_v = None

    def __call__(self, data: HeteroData) -> HeteroData:
        for fs in [self.oracle_drop_c, self.drop_c, self.add_c, self.scale_c, self.scale_v, self.add_v]:
            if fs is not None:
                data = fs(data)
        return data


class ComboInterpolateTransforms(ComboPreservedTransforms):
    def __init__(self, tf_dict: Dict, num_samples: int):
        super().__init__(tf_dict)
        self.tf_dict = tf_dict
        self.num_samples = num_samples

    def __call__(self, data: HeteroData) -> HeteroData:
        tf_list = [self.oracle_drop_c, self.drop_c, self.add_c, self.scale_c, self.scale_v, self.add_v]
        if self.num_samples == -1:
            selected_idx = np.arange(len(tf_list))
        else:
            selected_idx = np.random.choice(len(tf_list), min(self.num_samples, len(tf_list)), replace=False)

        for i, fs in enumerate(tf_list):
            if fs is not None and i in selected_idx:
                max_p = self.tf_dict[str(fs)]
                fs.p = random.random() * max_p
                data = fs(data)
        return data
