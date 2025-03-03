import os
import pdb
import time

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torch_geometric.utils import scatter
from torch_sparse import SparseTensor
from torch_sparse import spmm


def sync_timer():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def save_run_config(args: DictConfig):
    if args.ckpt:
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        prefix = f'{args.wandb.project}_{args.wandb.name}'
        exist_runs = [d for d in os.listdir('logs') if d.startswith(prefix)]
        log_folder_name = f'logs/{prefix}_exp{len(exist_runs)}'
        os.mkdir(log_folder_name)
        # with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
        #     yaml.dump(vars(args), outfile, default_flow_style=False)
        OmegaConf.save(args, os.path.join(log_folder_name, 'config.yaml'))
        return log_folder_name
    return None


def recover_lp_from_data(data, dtype=np.float32):
    data = data.to('cpu')
    c = data.q.numpy().astype(dtype)
    b = data.b.numpy().astype(dtype)
    A = SparseTensor(row=data['cons', 'to', 'vals'].edge_index[0],
                     col=data['cons', 'to', 'vals'].edge_index[1],
                     value=data['cons', 'to', 'vals'].edge_attr.squeeze(),
                     sparse_sizes=(data['cons'].num_nodes, data['vals'].num_nodes),
                     is_sorted=True, trust_data=True).to_dense().numpy().astype(dtype)
    # todo: might vary
    lb = np.zeros(A.shape[1]).astype(dtype)
    ub = None
    return A, c, b, lb, ub


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
