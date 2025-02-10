import torch
from torch_geometric.nn import MLP

from models.hetero_encoder import TripartiteHeteroEncoder


class TripartiteHeteroPretrainGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 head,
                 concat,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 norm):
        super().__init__()

        self.encoder = TripartiteHeteroEncoder(
            conv,
            head,
            concat,
            hid_dim,
            num_encode_layers,
            num_conv_layers,
            num_mlp_layers,
            num_pred_layers,
            norm
        )

        self.predictor = MLP([hid_dim, hid_dim * 2, hid_dim], norm=None)

    def forward(self, data):
        obj_embedding = self.encoder(data)
        embedding = self.predictor(obj_embedding)
        return embedding
