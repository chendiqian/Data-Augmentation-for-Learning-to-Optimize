import copy

import torch
from torch.nn import functional as F
from torch_geometric.nn import MLP

from models.backbone import Backbone
from utils.models import EMA, update_moving_average


class IGSDPretrainGNN(torch.nn.Module):
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

        # their default
        self.ema_updater = EMA(0.99)

        self.encoder1 = Backbone(
            is_qp,
            conv,
            hid_dim,
            num_encode_layers,
            num_conv_layers,
            num_mlp_layers,
            backbone_pred_layers,
            norm,
        )
        self.encoder2 = copy.deepcopy(self.encoder1)

        assert num_pred_layers > 0  # IGSD must have a predictor
        self.predictor = MLP([hid_dim] * (num_pred_layers + 1), norm=None)

    def forward(self, data, ppr_data):
        # encoder + projector
        online_embedding, *_ = self.encoder1(data)
        with torch.no_grad():
            # encoder + projector
            target_embedding, *_ = self.encoder2(ppr_data)
            # update weights
            update_moving_average(self.ema_updater, self.encoder2, self.encoder1)

        online_z = self.predictor(online_embedding)
        target_z = self.predictor(target_embedding)

        online_embedding = F.normalize(online_embedding, dim=-1, p=2)
        target_embedding = F.normalize(target_embedding, dim=-1, p=2)

        consistence_mat = torch.norm(online_z[:, None, :] - target_embedding[None, :, :], dim=-1, p=2) + \
                          torch.norm(target_z[:, None, :] - online_embedding[None, :, :], dim=-1, p=2)

        return consistence_mat
