from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.nn import functional as F
from torch_scatter import scatter


class Loss(ABC):
    @abstractmethod
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss


class JSD(Loss):
    def __init__(self, discriminator=lambda x, y: x @ y.t()):
        super(JSD, self).__init__()
        self.discriminator = discriminator

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        num_neg = neg_mask.int().sum()
        num_pos = pos_mask.int().sum()
        similarity = self.discriminator(anchor, sample)

        E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
        E_pos /= num_pos

        neg_sim = similarity * neg_mask
        E_neg = (F.softplus(- neg_sim) + neg_sim - np.log(2)).sum()
        E_neg /= num_neg

        return E_neg - E_pos


def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask


class Sampler(ABC):
    def __init__(self, intraview_negs=False):
        self.intraview_negs = intraview_negs

    def __call__(self, anchor, sample, *args, **kwargs):
        ret = self.sample(anchor, sample, *args, **kwargs)
        if self.intraview_negs:
            ret = self.add_intraview_negs(*ret)
        return ret

    @abstractmethod
    def sample(self, anchor, sample, *args, **kwargs):
        pass

    @staticmethod
    def add_intraview_negs(anchor, sample, pos_mask, neg_mask):
        num_nodes = anchor.size(0)
        device = anchor.device
        intraview_pos_mask = torch.zeros_like(pos_mask, device=device)
        intraview_neg_mask = torch.ones_like(pos_mask, device=device) - torch.eye(num_nodes, device=device)
        new_sample = torch.cat([sample, anchor], dim=0)                     # (M+N) * K
        new_pos_mask = torch.cat([pos_mask, intraview_pos_mask], dim=1)     # M * (M+N)
        new_neg_mask = torch.cat([neg_mask, intraview_neg_mask], dim=1)     # M * (M+N)
        return anchor, new_sample, new_pos_mask, new_neg_mask


class CrossScaleSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(CrossScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, batch=None, neg_sample=None, *args, **kwargs):
        num_graphs = anchor.shape[0]  # M
        num_nodes = sample.shape[0]   # N
        device = sample.device

        if neg_sample is not None:
            assert num_graphs == 1  # only one graph, explicit negative samples are needed
            assert sample.shape == neg_sample.shape
            pos_mask1 = torch.ones((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask0 = torch.zeros((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask = torch.cat([pos_mask1, pos_mask0], dim=1)     # M * 2N
            sample = torch.cat([sample, neg_sample], dim=0)         # 2N * K
        else:
            assert batch is not None
            ones = torch.eye(num_nodes, dtype=torch.float32, device=device)     # N * N
            pos_mask = scatter(ones, batch, dim=0, reduce='sum')                # M * N

        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask
