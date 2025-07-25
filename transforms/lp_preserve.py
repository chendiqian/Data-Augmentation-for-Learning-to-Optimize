import math
from typing import Dict, Tuple
import random

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph
from sklearn.datasets import make_sparse_spd_matrix
from scipy.sparse import random_array, eye_array, csr_array, vstack as sp_vstack

from utils.evaluation import is_qp, oracle_inactive_constraints, inactive_contraint_heuristic


def drop_var(data: HeteroData, drop_idx: np.ndarray, remain_vars: np.ndarray):
    c2v_edge_index = data[('cons', 'to', 'vals')].edge_index.numpy()
    keep_edge_mask = ~np.isin(c2v_edge_index[1], drop_idx)  # we drop columns of A matrix

    c2v_edge_index = c2v_edge_index[:, keep_edge_mask]
    _, remapped_a = np.unique(c2v_edge_index[1], return_inverse=True)
    c2v_edge_index[1] = remapped_a

    c2v_edge_index = torch.from_numpy(c2v_edge_index).long()
    c2v_edge_attr = data[('cons', 'to', 'vals')].edge_attr[keep_edge_mask]

    if is_qp(data):
        v2v_edge_index, v2v_edge_attr = subgraph(subset=torch.from_numpy(remain_vars),
                                                 edge_index=data[('vals', 'to', 'vals')].edge_index,
                                                 edge_attr=data[('vals', 'to', 'vals')].edge_attr,
                                                 relabel_nodes=True,
                                                 num_nodes=data['vals'].num_nodes)
    else:
        v2v_edge_index, v2v_edge_attr = None, None

    return c2v_edge_index, c2v_edge_attr, v2v_edge_index, v2v_edge_attr


def drop_cons(data: HeteroData, drop_idx: np.ndarray) -> Tuple[torch.Tensor, torch.FloatTensor]:
    edge_index = data[('cons', 'to', 'vals')].edge_index.numpy()
    keep_edge_mask = ~np.isin(edge_index[0], drop_idx)

    edge_index = edge_index[:, keep_edge_mask]
    _, remapped_a = np.unique(edge_index[0], return_inverse=True)
    edge_index[0] = remapped_a

    new_edge_index = torch.from_numpy(edge_index).long()
    new_edge_attr = data[('cons', 'to', 'vals')].edge_attr[keep_edge_mask]
    return new_edge_index, new_edge_attr


class OracleBiasProblem:
    def __init__(self, strength=0.1):
        assert strength > 0.
        self.p = strength

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        _is_qp = is_qp(data)

        solution = data.x_solution.numpy()

        if _is_qp:
            Q_dens = data[('vals', 'to', 'vals')].edge_index.shape[1] / (n ** 2)
            Q_bias = make_sparse_spd_matrix(n_dim=n, alpha=1 - Q_dens * self.p / 2.,
                                            smallest_coef=0.1, largest_coef=0.9, sparse_format='coo')
            Q_bias /= Q_bias.max()
            extra_v2v_edge_index = torch.from_numpy(np.vstack([Q_bias.row, Q_bias.col])).long()
            extra_v2v_edge_attr = torch.from_numpy(Q_bias.data).float()[:, None]
            Qx = Q_bias @ solution
        else:
            Qx = np.zeros_like(solution)

        A_dens = data[('cons', 'to', 'vals')].edge_index.shape[1] / (m * n)
        # otherwise it is too negative obj
        A_bias = -random_array((m, n), density=A_dens, format='coo')
        extra_c2v_edge_index = torch.from_numpy(np.vstack([A_bias.row, A_bias.col])).long()
        extra_c2v_edge_attr = torch.from_numpy(A_bias.data).float()[:, None]

        q_bias = A_bias.T @ data.duals.numpy() - Qx
        new_q = data.q.numpy() + q_bias
        new_b = data.b.numpy() + A_bias @ solution

        obj = data.obj_solution + 0.5 * solution @ Qx + q_bias @ solution

        # we don't have to coalesce, as we use add aggregation in message passing
        new_data = data.__class__(
            cons={
                'num_nodes': m,
                'x': data['cons'].x,
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            cons__to__vals={'edge_index': torch.hstack([data[('cons', 'to', 'vals')].edge_index,
                                                        extra_c2v_edge_index]),
                            'edge_attr': torch.vstack([data[('cons', 'to', 'vals')].edge_attr,
                                                       extra_c2v_edge_attr])},
            q=torch.from_numpy(new_q).float(),
            b=torch.from_numpy(new_b).float(),
            obj_solution=obj,
            x_solution=data.x_solution,
            duals=data.duals,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )

        if _is_qp:
            new_data[('vals', 'to', 'vals')].edge_index = torch.hstack([data[('vals', 'to', 'vals')].edge_index,
                                                                        extra_v2v_edge_index])
            new_data[('vals', 'to', 'vals')].edge_attr = torch.vstack([data[('vals', 'to', 'vals')].edge_attr,
                                                                       extra_v2v_edge_attr])

        return new_data

    def __repr__(self):
        return "OracleBiasProblem"


class OracleDropIdleVariable:
    """
    Drop x where x=0
    """

    def __init__(self, strength=0.1):
        assert 0 < strength < 1
        self.p = strength

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        idle_idx = torch.where(data.x_solution.abs() < 1.e-7)[0].numpy()

        drop_idx = np.random.choice(idle_idx, size=min(int(n * self.p), len(idle_idx)), replace=False)
        remain_vars = ~np.isin(np.arange(n), drop_idx)
        c2v_edge_index, c2v_edge_attr, v2v_edge_index, v2v_edge_attr = drop_var(data, drop_idx, remain_vars)

        new_data = data.__class__(
            cons={
                'num_nodes': m,
                'x': data['cons'].x,
            },
            vals={
                'num_nodes': remain_vars.sum().item(),
                'x': data['vals'].x[remain_vars],
            },
            cons__to__vals={'edge_index': c2v_edge_index,
                            'edge_attr': c2v_edge_attr},
            q=data.q[remain_vars],
            b=data.b,
            obj_solution=data.obj_solution,
            x_solution=data.x_solution[remain_vars],
            duals=data.duals,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )

        if is_qp(data):
            new_data[('vals', 'to', 'vals')].edge_index = v2v_edge_index
            new_data[('vals', 'to', 'vals')].edge_attr = v2v_edge_attr
        return new_data

    def __repr__(self):
        return 'OracleDropIdleVariable'


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
            duals=data.duals,
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
            duals=data.duals,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )

        if is_qp(data):
            new_data[('vals', 'to', 'vals')].edge_index = data[('vals', 'to', 'vals')].edge_index
            new_data[('vals', 'to', 'vals')].edge_attr = data[('vals', 'to', 'vals')].edge_attr
        return new_data

    def __repr__(self):
        return 'DropInactiveConstraint'


class AddRedundantConstraint:
    """
    Add more constraints PAx <= Pb + eps.
    """

    def __init__(self, strength=0.2, affinity=3, bias=True, use_sparse=False):
        assert 0 < strength < 1
        self.p = strength
        self.affinity = affinity
        self.bias = bias
        self.use_sparse = use_sparse

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        num_new_cons = int(m * self.p)

        edge_index = data[('cons', 'to', 'vals')].edge_index.numpy()
        edge_attr = data[('cons', 'to', 'vals')].edge_attr.squeeze(1).numpy()

        if self.use_sparse:
            eye_mat = eye_array(m, format='csr')
            rows = np.arange(num_new_cons).repeat(self.affinity)
            cols = np.random.randint(low=0, high=m, size=self.affinity * num_new_cons)
            values = np.random.rand(self.affinity * num_new_cons)
            rand_mat = csr_array((values, (rows, cols)), shape=(num_new_cons, m))

            mat = sp_vstack([eye_mat, rand_mat])

            A = csr_array((edge_attr, (edge_index[0], edge_index[1])), shape=(m, n))
            A_new = (mat @ A).tocoo()
            edge_index = torch.from_numpy(np.vstack([A_new.row, A_new.col])).long()
            edge_attr = torch.from_numpy(A_new.data)[:, None].float()

            new_b = mat @ data.b.numpy()
            if self.bias:
                bias = np.random.randn(new_b.shape[0])
                bias[bias < 0] = 0.
                bias[:m] = 0
                new_b += bias
            new_b = torch.from_numpy(new_b).float()
        else:
            idx = np.random.choice(m, (num_new_cons, self.affinity), replace=True)
            weights = np.random.rand(*idx.shape)

            A = np.zeros((m, n), dtype=np.float32)
            A[edge_index[0], edge_index[1]] = edge_attr

            extra_A = np.einsum('nk,nkf->nf', weights, A[idx])
            where = np.where(extra_A)
            extra_edge_index = torch.from_numpy(np.vstack(where)).long()
            extra_edge_index[0] += m
            extra_edge_attr = torch.from_numpy(extra_A[where[0], where[1]])[:, None].float()

            edge_index = torch.hstack([data[('cons', 'to', 'vals')].edge_index, extra_edge_index])
            edge_attr = torch.vstack([data[('cons', 'to', 'vals')].edge_attr, extra_edge_attr])

            new_b = (weights * data.b.numpy()[idx]).sum(1)
            if self.bias:
                bias = np.random.randn(new_b.shape[0])
                bias[bias < 0] = 0.
                new_b += bias

            new_b = torch.from_numpy(new_b).float()
            new_b = torch.hstack([data.b, new_b])

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
            x_solution=data.x_solution,
            duals=data.duals,
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
            duals=data.duals,
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
            duals=data.duals,
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
            duals=data.duals,
            inactive_idx=data.inactive_idx,
            heur_idx=data.heur_idx,
        )

        if is_qp(data):
            # theoretically, if we augment Q to [Q, A.T; A, B] and maintain PSD, it should be B - A.T @ Q_inv @ A >= 0
            # so we just augment with a diagonal B and A being zeros
            row = col = torch.arange(num_new_vars) + n
            extra_v2v_edge_index = torch.vstack([row, col]).long()
            extra_v2v_edge_attr = torch.rand(num_new_vars)

            new_data[('vals', 'to', 'vals')].edge_index = torch.hstack([data[('vals', 'to', 'vals')].edge_index,
                                                                        extra_v2v_edge_index])
            new_data[('vals', 'to', 'vals')].edge_attr = torch.vstack([data[('vals', 'to', 'vals')].edge_attr,
                                                                       extra_v2v_edge_attr[:, None]])
        return new_data

    def __repr__(self):
        return 'AddDumbVariables'


class ComboPreservedTransforms:
    def __init__(self, tf_dict: Dict, num_samples: int, interpolate: bool = False):
        self.tf_dict = tf_dict
        strengths = tf_dict.values()
        assert max(strengths) > 0, "At least 1 transformation!"

        assert not ('OracleDropInactiveConstraint' in tf_dict and
                    'DropInactiveConstraint' in tf_dict and
                    tf_dict['OracleDropInactiveConstraint'] * tf_dict['DropInactiveConstraint'] > 0), "Cannot both"

        if 'OracleDropIdleVariable' in tf_dict and tf_dict['OracleDropIdleVariable'] > 0:
            self.oracle_drop_v = OracleDropIdleVariable(tf_dict['OracleDropIdleVariable'])
        else:
            self.oracle_drop_v = None

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

        self.num_samples = num_samples
        self.interpolate = interpolate

    def __call__(self, data: HeteroData) -> HeteroData:
        full_tf_list = [self.oracle_drop_v,
                        self.oracle_drop_c, self.drop_c,
                        self.add_c, self.scale_c,
                        self.scale_v, self.add_v]
        tf_list = [tf for tf in full_tf_list if tf is not None]
        assert len(tf_list)

        if self.num_samples == -1 or self.num_samples >= len(tf_list):
            selected_idx = np.arange(len(tf_list))
        else:
            selected_idx = np.random.choice(len(tf_list), self.num_samples, replace=False)
            selected_idx = np.sort(selected_idx)

        for i in selected_idx:
            fs = tf_list[i]

            if self.interpolate:
                max_p = self.tf_dict[str(fs)]
                fs.p = random.random() * max_p

            data = fs(data)
        return data
