import torch
from torch_geometric.nn import Linear, MLP

from mol_models.encoder import Encoder


class BasicGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_backbone_mlp,
                 num_mlp_layers,
                 norm):
        super().__init__()

        self.encoder = Encoder(
            conv,
            hid_dim,
            num_conv_layers,
            num_mlp_layers,
            num_backbone_mlp,
            norm
        )

        self.predictor = MLP([hid_dim] * num_pred_layers + [1])

    def forward(self, data):
        obj_pred = self.encoder(data)
        x = self.predictor(obj_pred)
        return x.squeeze()


class PretrainGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_backbone_mlp,
                 num_mlp_layers,
                 norm):
        super().__init__()

        self.encoder = Encoder(
            conv,
            hid_dim,
            num_conv_layers,
            num_mlp_layers,
            num_backbone_mlp,
            norm
        )

        if num_pred_layers > 0:
            self.predictor = MLP([hid_dim] * (num_pred_layers + 1), norm=None)
        else:
            self.predictor = torch.nn.Identity()

    def forward(self, data):
        obj_pred = self.encoder(data)
        x = self.predictor(obj_pred)
        return x
