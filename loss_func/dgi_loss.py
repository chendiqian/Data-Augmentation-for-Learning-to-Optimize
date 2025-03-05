import torch

from loss_func.pygcl_utils import add_extra_mask, CrossScaleSampler


class SingleBranchContrast(torch.nn.Module):
    def __init__(self, loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(SingleBranchContrast, self).__init__()
        assert mode == 'G2L'  # only global-local pairs allowed in single-branch contrastive learning
        self.loss = loss
        self.mode = mode
        self.sampler = CrossScaleSampler(intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h, g, batch=None, extra_pos_mask=None, extra_neg_mask=None):
        assert batch is not None
        anchor, sample, pos_mask, neg_mask = self.sampler(anchor=g, sample=h, batch=batch)

        pos_mask, neg_mask = add_extra_mask(pos_mask, neg_mask, extra_pos_mask, extra_neg_mask)
        loss = self.loss(anchor=anchor, sample=sample, pos_mask=pos_mask, neg_mask=neg_mask, **self.kwargs)
        return loss
