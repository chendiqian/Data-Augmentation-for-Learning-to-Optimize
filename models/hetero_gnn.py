import torch
from torch_geometric.nn import MLP

from models.hetero_encoder import BipartiteHeteroEncoder, TripartiteHeteroEncoder


class BipartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 head,
                 concat,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 hid_pred,
                 num_mlp_layers,
                 norm):
        super().__init__()

        self.encoder = BipartiteHeteroEncoder(
            conv,
            head,
            concat,
            hid_dim,
            num_encode_layers,
            num_conv_layers,
            num_mlp_layers,
            norm)

        if hid_pred == -1:
            hid_pred = hid_dim
        self.predictor = MLP([hid_dim] + [hid_pred] * num_pred_layers + [1], norm=None)

    def forward(self, data):
        x_dict = self.encoder(data)
        x = self.predictor(x_dict['vals'])
        return x.squeeze()


class TripartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 head,
                 concat,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 hid_pred,
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
            norm
        )

        if hid_pred == -1:
            hid_pred = hid_dim
        self.predictor = MLP([hid_dim] + [hid_pred] * num_pred_layers + [1], norm=None)

    def forward(self, data):
        x_dict = self.encoder(data)
        x = self.predictor(x_dict['vals'])
        return x.squeeze()
