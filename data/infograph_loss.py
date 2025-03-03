import math

import torch
import torch.nn.functional as F


# https://github.com/sunfanyunn/InfoGraph/blob/master/unsupervised/cortex_DIM/functions/gan_losses.py
def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)
    if measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    else:
        raise NotImplementedError

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    else:
        raise NotImplementedError

    if average:
        return Eq.mean()
    else:
        return Eq


# https://github.com/sunfanyunn/InfoGraph/blob/master/unsupervised/losses.py
def local_global_loss(l_enc, g_enc, nnodes, measure='JSD'):
    """

    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    """
    device = l_enc.device
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    batch = torch.arange(num_graphs, device=device).repeat_interleave(nnodes)

    pos_mask = torch.zeros(num_nodes, num_graphs, dtype=torch.float, device=device)
    neg_mask = torch.ones(num_nodes, num_graphs, dtype=torch.float, device=device)
    arange_nnodes = torch.arange(num_nodes, device=device)

    pos_mask[arange_nnodes, batch] = 1.
    neg_mask[arange_nnodes, batch] = 0.

    res = torch.mm(l_enc, g_enc.t())   # nnodes x ngraphs

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos
