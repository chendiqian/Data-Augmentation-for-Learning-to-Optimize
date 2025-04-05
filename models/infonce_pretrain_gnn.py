import torch
from torch_geometric.nn import MLP

from models.backbone import Backbone


class PretrainGNN(torch.nn.Module):
    def __init__(self,
                 is_qp,
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
            is_qp,
            conv,
            hid_dim,
            num_encode_layers,
            num_conv_layers,
            num_mlp_layers,
            backbone_pred_layers,
            norm,
        )

        if num_pred_layers == 0:
            self.predictor = torch.nn.Identity()
        else:
            self.predictor = MLP([hid_dim] * (num_pred_layers + 1), norm=None)

    def forward(self, data):
        obj_embedding, *_ = self.encoder(data)
        obj_embedding = self.predictor(obj_embedding)
        return obj_embedding, None
