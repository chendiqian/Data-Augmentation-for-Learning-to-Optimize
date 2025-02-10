import torch
from torch_geometric.nn import Linear

from mol_models.encoder import Encoder


class BasicGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 norm):
        super().__init__()

        self.encoder = Encoder(
            conv,
            hid_dim,
            num_conv_layers,
            num_mlp_layers,
            num_pred_layers,
            norm
        )

        self.predictor = Linear(hid_dim, 1)

    def forward(self, data):
        obj_pred = self.encoder(data)
        x = self.predictor(obj_pred)
        return x.squeeze()
