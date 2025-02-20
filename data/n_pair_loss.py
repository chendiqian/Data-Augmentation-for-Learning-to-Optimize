import torch
import torch.nn.functional as F
from torch import nn


# similar to paired version of
# https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py
class NPairLoss(nn.Module):
    """
    https://papers.nips.cc/paper_files/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (N, M, D): Tensor with negative samples (e.g. embeddings of other inputs)

    Returns:
         Value of the Loss.
    """

    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query, positive_key, negative_keys):
        # Normalize to unit vectors
        query = F.normalize(query, p=2, dim=-1)
        positive_key = F.normalize(positive_key, p=2, dim=-1)
        negative_keys = F.normalize(negative_keys, p=2, dim=-1)

        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)  # N x 1
        negative_logits = torch.einsum('bf,bnf->bn', query, negative_keys)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)

        return F.cross_entropy(logits / self.temperature, labels, reduction=self.reduction)
