import torch
from torch_geometric.nn import MLP
from models.backbone import Backbone


class GNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 backbone_pred_layers,
                 norm):
        super().__init__()

        self.encoder = Backbone(
            conv,
            hid_dim,
            num_encode_layers,
            num_conv_layers,
            num_mlp_layers,
            backbone_pred_layers,
            norm,
            output_nodes=False  # we don't need node embeddings
        )

        self.predictor = MLP([hid_dim] * num_pred_layers + [1], norm=None)

    def forward(self, data):
        obj_pred, *_ = self.encoder(data)
        x = self.predictor(obj_pred)
        return x.squeeze()
