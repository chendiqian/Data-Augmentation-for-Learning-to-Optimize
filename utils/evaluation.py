import numpy as np
import torch
from scipy import sparse as sp
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter
from torch_scatter import scatter_sum
from torch_sparse import SparseTensor, spmm


def recover_lp_from_data(data, dtype=np.float32):
    data = data.to('cpu')
    c = data.q.numpy().astype(dtype)
    b = data.b.numpy().astype(dtype)
    A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
                     col=data['cons', 'to', 'vals'].edge_index[1],
                     value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
                     sparse_sizes=(data['cons'].num_nodes, data['vals'].num_nodes)).to_dense().numpy().astype(dtype)
    # todo: might vary
    lb = np.zeros(A.shape[1]).astype(dtype)
    ub = None
    return A, c, b, lb, ub


def recover_qp_from_data(data, dtype=np.float32):
    data = data.to('cpu')
    c = data.q.numpy().astype(dtype)
    b = data.b.numpy().astype(dtype)
    A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
                     col=data['cons', 'to', 'vals'].edge_index[1],
                     value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
                     sparse_sizes=(data['cons'].num_nodes, data['vals'].num_nodes)).to_dense().numpy().astype(dtype)
    P = SparseTensor(row=data['vals', 'to', 'vals'].edge_index[0],
                     col=data['vals', 'to', 'vals'].edge_index[1],
                     value=data['vals', 'to', 'vals'].edge_attr.squeeze(),
                     sparse_sizes=(data['vals'].num_nodes, data['vals'].num_nodes)).to_dense().numpy().astype(dtype)
    # todo: might vary
    lb = np.zeros(A.shape[1]).astype(dtype)
    ub = None
    return P, A, c, b, lb, ub


def normalize_cons(A, b):
    if A is None or b is None:
        return A, b
    Ab = np.concatenate([A, b[:, None]], axis=1)
    max_logit = np.abs(Ab).max(axis=1)
    max_logit[max_logit == 0] = 1.
    Ab = Ab / max_logit[:, None]
    A = Ab[:, :-1]
    b = Ab[:, -1]
    return A, b


def calc_violation(pred, data):
    assert pred.dim() <= 2
    if pred.dim() == 2:
        assert pred.shape[1] == 1
    if pred.dim() == 1:
        pred = pred[:, None]
    Ax_minus_b = spmm(data['cons', 'to', 'vals'].edge_index,
                      data['cons', 'to', 'vals'].edge_attr.squeeze(),
                      data['cons'].num_nodes, data['vals'].num_nodes, pred).squeeze() - data.b
    violation = scatter(torch.relu(Ax_minus_b), data['cons'].batch, dim=0, reduce='mean')  # (batchsize,)
    return violation


def compute_acc(pred1, pred2):
    """
    for evaluating SSL

    Args:
        pred1:
        pred2:

    Returns:

    """
    # # top 5 acc
    # cos = F.cosine_similarity(pred1.detach()[:, None], pred2.detach()[None, :], dim=-1)
    # pos_mask = torch.eye(cos.shape[0], device=pred1.device).bool()
    # comb_sim = torch.cat([cos[pos_mask][:, None], cos.masked_fill_(pos_mask, -1e10)], dim=1)
    # sim_argsort = comb_sim.argsort(dim=1, descending=True).argmin(dim=-1)
    # num_corrects = (sim_argsort < 5).sum()

    # acc: pos-pair embedding has the highest cos similarity
    cos = F.cosine_similarity(pred1.detach()[:, None], pred2.detach()[None, :], dim=-1)
    sort_idx = cos.argmax(dim=1)
    sort_label = torch.arange(pred1.shape[0], device=pred1.device)
    num_corrects = (sort_idx == sort_label).sum()
    return num_corrects


def numpy_inactive_contraint_heuristic(A, b, c,):
    """
    A heuristic to give the likely inactive constraints. A @ c + b, the smaller, the more likely.
    An assumption is, for a constraint matrix with shape (m, n) m > n, there are exactly n active ones.

    Args:
        A:
        b:
        c:

    Returns:

    """
    m, n = A.shape
    c_normed = c / np.linalg.norm(c, axis=0)
    Anorms = np.linalg.norm(A, axis=1, keepdims=True)
    A_normed = A / Anorms

    A_full = np.vstack([A_normed, -np.eye(n)])
    b_full = np.hstack([b / Anorms.squeeze(), np.zeros(n)])

    heur = A_full @ c_normed + b_full

    heur_idx = np.argsort(heur)
    heur_idx = heur_idx[n:]
    heur_idx = heur_idx[heur_idx < m]

    return heur_idx


def numpy_inactive_contraint(A, b, x, eps=1.e-6):
    """
    Return the inactive constraints where A@x < b strictly holds

    Args:
        A:
        b:
        x:
        eps:

    Returns:

    """
    return np.where(~(np.abs(A @ x - b) < eps))[0]


def inactive_contraint_heuristic(c2v_edge_index, c2v_edge_attr, b, c, m, n):
    """
    A torch version.

    Args:
        c2v_edge_index:
        c2v_edge_attr:
        b:
        c:
        m:
        n:

    Returns:

    """
    raise DeprecationWarning

    row, col = c2v_edge_index
    values = c2v_edge_attr

    Anorms = scatter_sum(values ** 2, row).numpy() ** 0.5
    c = c.numpy()
    c = c / np.linalg.norm(c)

    row, col = row.numpy(), col.numpy()
    values = values.numpy()

    # we assume each row, col is selected at least once
    A = sp.csr_array((values / Anorms[row], (row, col)), shape=(m, n))
    A = sp.vstack([A, -sp.eye_array(n, format='csr')])
    b = np.hstack([b.numpy() / Anorms, np.zeros(n)])

    heur = A @ c + b

    heur_idx = np.argsort(heur)
    heur_idx = heur_idx[n:]
    heur_idx = heur_idx[heur_idx < m]

    return heur_idx


def oracle_inactive_constraints(solution, c2v_edge_index, c2v_edge_attr, b, eps=1.e-6):
    """
    A torch version.

    Args:
        solution:
        c2v_edge_index:
        c2v_edge_attr:
        b:
        eps:

    Returns:

    """
    raise DeprecationWarning

    violations = scatter_sum(c2v_edge_attr * solution[c2v_edge_index[1]], c2v_edge_index[0]) - b
    active_mask = violations.abs() < eps
    inactive_mask = ~active_mask
    inactive_idx = torch.where(inactive_mask)[0]
    return inactive_idx


def is_qp(data: HeteroData):
    return ('vals', 'to', 'vals') in data.edge_index_dict
