import numpy as np
import torch
from scipy.linalg import inv
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_adj, to_undirected

from utils.evaluation import is_qp


def c2v_ppr(data: HeteroData, alpha: float):
    m, n = data['cons'].num_nodes, data['vals'].num_nodes
    ne = data['cons', 'to', 'vals'].edge_index.shape[1]

    homo_edge_index = data['cons', 'to', 'vals'].edge_index.clone()  # CANNOT IN-SPACE!!!!
    homo_edge_index[1] += m
    homo_edge_index = to_undirected(homo_edge_index, num_nodes=m + n)
    A = to_dense_adj(edge_index=homo_edge_index).squeeze().numpy()
    deg = np.sum(A, 1)
    d_inv = deg ** -0.5
    d_inv[np.isinf(d_inv)] = 1.
    d_inv[np.isnan(d_inv)] = 1.
    A = A * d_inv[None] * d_inv[:, None]

    diffusion = alpha * inv((np.eye(A.shape[0]) - (1 - alpha) * A))
    row, col = homo_edge_index.numpy()
    # we don't want to select existing edges
    diffusion[row, col] = -1.e10
    # we take the original bipartite part
    # does not make sense to add intra partition edges
    diffusion = diffusion[:m, m:].reshape(-1)
    sort_idx = np.argsort(diffusion, axis=None)
    diffusion = diffusion[sort_idx]
    inf_mask = diffusion == -1.e10
    sort_idx = sort_idx[~inf_mask]
    diffusion = diffusion[~inf_mask]
    sort_idx = torch.from_numpy(sort_idx).long()[-ne:]
    diffusion = torch.from_numpy(diffusion).float()[-ne:]
    return sort_idx, diffusion


def v2v_ppr(data: HeteroData, alpha: float):
    ne = data['vals', 'to', 'vals'].edge_index.shape[1]

    A = to_dense_adj(edge_index=data['vals', 'to', 'vals'].edge_index).squeeze().numpy()
    deg = np.sum(A, 1)
    d_inv = deg ** -0.5
    d_inv[np.isinf(d_inv)] = 1.
    d_inv[np.isnan(d_inv)] = 1.
    A = A * d_inv[None] * d_inv[:, None]

    diffusion = alpha * inv((np.eye(A.shape[0]) - (1 - alpha) * A))
    row, col = data['vals', 'to', 'vals'].edge_index.numpy()
    # we don't want to select existing edges
    diffusion[row, col] = -1.e10
    diffusion = diffusion.reshape(-1)
    sort_idx = np.argsort(diffusion, axis=None)
    diffusion = diffusion[sort_idx]
    inf_mask = diffusion == -1.e10
    sort_idx = sort_idx[~inf_mask]
    diffusion = diffusion[~inf_mask]
    sort_idx = torch.from_numpy(sort_idx).long()[-ne:]
    diffusion = torch.from_numpy(diffusion).float()[-ne:]
    return sort_idx, diffusion


class PageRankAugment:
    """
    PPR graph diffusion, for IGSD the distillation method, and MVGRL
    """
    def __init__(self, strength=0.3):
        self.strength = strength
        self.alpha = 0.2

    def __call__(self, data: HeteroData) -> HeteroData:
        # we allow no augmentation!
        if self.strength == 0.:
            return data

        m, n = data['cons'].num_nodes, data['vals'].num_nodes
        ne = data['cons', 'to', 'vals'].edge_index.shape[1]

        if hasattr(data, 'ppr_scores') and hasattr(data, 'ppr_idx'):
            diffusion = data.ppr_scores
            sort_idx = data.ppr_idx
        else:
            sort_idx, diffusion = c2v_ppr(data, self.alpha)

        num_new_edges = int(self.strength * ne)
        idx = sort_idx[-num_new_edges:]

        new_row, new_col = idx // n, idx % n
        extra_edge_index = torch.vstack([new_row, new_col]).long()
        # todo: no better ideas how to generate edge attr for the edges
        extra_edge_attr = diffusion[-num_new_edges:].float()[:, None]

        new_data = data.__class__(
            cons={
                'num_nodes': data['cons'].num_nodes,
                'x': data['cons'].x,
            },
            vals={
                'num_nodes': data['vals'].num_nodes,
                'x': data['vals'].x,
            },
            cons__to__vals={'edge_index': torch.hstack([data['cons', 'to', 'vals'].edge_index, extra_edge_index]),
                            'edge_attr': torch.vstack([data['cons', 'to', 'vals'].edge_attr, extra_edge_attr])},
            q=data.q,
            b=data.b,
            obj_solution=data.obj_solution,  # this is actually not correct
        )

        if is_qp(data):
            ne = data['vals', 'to', 'vals'].edge_index.shape[1]

            if hasattr(data, 'qp_ppr_scores') and hasattr(data, 'qp_ppr_idx'):
                diffusion = data.qp_ppr_scores
                sort_idx = data.qp_ppr_idx
            else:
                sort_idx, diffusion = v2v_ppr(data, self.alpha)

            num_new_edges = int(self.strength * ne)
            idx = sort_idx[-num_new_edges:]

            new_row, new_col = idx // n, idx % n
            extra_edge_index = torch.vstack([new_row, new_col]).long()
            # todo: no better ideas how to generate edge attr for the edges
            extra_edge_attr = diffusion[-num_new_edges:].float()[:, None]

            new_data[('vals', 'to', 'vals')].edge_index = torch.hstack([data['vals', 'to', 'vals'].edge_index, extra_edge_index])
            new_data[('vals', 'to', 'vals')].edge_attr = torch.vstack([data['vals', 'to', 'vals'].edge_attr, extra_edge_attr])

        return new_data
