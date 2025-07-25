from typing import List, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter
from torch_sparse import SparseTensor, spmm


def gurobi_solve_lp(A, b, c, lb=0., ub=float('inf')):
    import gurobipy as gp
    from gurobipy import GRB

    m, n = A.shape
    model = gp.Model("lp")
    model.Params.LogToConsole = 0
    variables = model.addMVar(n, lb=lb, ub=ub)

    # Objective: 0.5 x^T P x + q^T x
    model.setObjective(c @ variables, GRB.MINIMIZE)

    # Add inequality constraints
    constrs = model.addConstr(A @ variables <= b)

    # Solve
    model.optimize()

    # Duals
    if model.status == GRB.OPTIMAL:
        duals = constrs.getAttr("Pi")
        solution = variables.X
    else:
        duals = solution = None
    return solution, duals, model


def gurobi_solve_qp(Q, c, Aub, bub, Aeq=None, beq=None, lb=0., ub=float('inf')):
    import gurobipy as gp
    from gurobipy import GRB

    _, n = Aub.shape
    model = gp.Model("qp")
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = 30
    variables = model.addMVar(n, lb=lb, ub=ub)

    # Objective: 0.5 x^T P x + q^T x
    model.setObjective(0.5 * variables @ Q @ variables + c @ variables)

    # Add inequality constraints
    constrs = model.addConstr(Aub @ variables <= bub)
    if Aeq is not None:
        constrs2 = model.addConstr(Aeq @ variables == beq)

    # Solve
    model.optimize()

    # Duals
    if model.status == GRB.OPTIMAL:
        duals = constrs.getAttr("Pi")
        if Aeq is not None:
            duals = np.hstack([duals, constrs2.getAttr("Pi")])
        solution = variables.X
    else:
        duals = solution = None
    return solution, duals, model


def recover_lp_from_data(data, dtype=np.float32):
    data = data.to('cpu')
    c = data.q.numpy().astype(dtype)
    b = data.b.numpy().astype(dtype)
    A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
                     col=data['cons', 'to', 'vals'].edge_index[1],
                     value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
                     sparse_sizes=(data['cons'].num_nodes, data['vals'].num_nodes)).to_scipy('csr').toarray().astype(dtype)
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
                     sparse_sizes=(data['cons'].num_nodes, data['vals'].num_nodes)).to_scipy('csr').toarray().astype(dtype)
    P = SparseTensor(row=data['vals', 'to', 'vals'].edge_index[0],
                     col=data['vals', 'to', 'vals'].edge_index[1],
                     value=data['vals', 'to', 'vals'].edge_attr.squeeze(),
                     sparse_sizes=(data['vals'].num_nodes, data['vals'].num_nodes)).to_scipy('csr').toarray().astype(dtype)
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


def inactive_contraint_heuristic(data):
    if is_qp(data):
        Q, A, c, b, *_ = recover_qp_from_data(data)
    else:
        A, c, b, *_ = recover_lp_from_data(data)
        Q = None

    return data_contraint_heuristic(Q, A, b, c)


def data_contraint_heuristic(Q, A, b, c):
    m, n = A.shape

    c_normed = c / np.linalg.norm(c, axis=0)
    Anorms = np.linalg.norm(A, axis=1, keepdims=True)
    A_normed = A / Anorms

    A_full = np.vstack([A_normed, -np.eye(n)])
    b_full = np.hstack([b / Anorms.squeeze(), np.zeros(n)])

    heur = A_full @ c_normed + b_full

    # Todo: in theory the dist should also be considered
    # if is_qp(data):
    #     # the global min of unconstrained programming
    #     xmin = np.linalg.solve(Q, -c)
    #     # the distance from the global min to the hyperplane
    #     dist = np.abs(A_full @ xmin - b_full)
    #
    # heur = heur + dist

    heur_idx = np.argsort(heur)
    heur_idx = heur_idx[n:]
    heur_idx = heur_idx[heur_idx < m]

    return heur_idx


def oracle_inactive_constraints(data, eps=1.e-6):
    # we don't need to recover Q matrix
    A, c, b, *_ = recover_lp_from_data(data)
    x = data.x_solution.numpy()
    return data_inactive_constraints(A, b, x, eps)


def data_inactive_constraints(A, b, solution, eps=1.e-6):
    return np.where(~(np.abs(A @ solution - b) < eps))[0]


def is_qp(data: HeteroData):
    if isinstance(data, HeteroData):
        d = data
    elif isinstance(data, (List, Tuple)):
        d = data[0]
    return ('vals', 'to', 'vals') in d.edge_index_dict
