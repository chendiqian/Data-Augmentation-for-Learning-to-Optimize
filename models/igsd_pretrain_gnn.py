import copy

import torch
from torch.nn import functional as F
from torch_geometric.nn import MLP

from models.backbone import Backbone


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class IGSDPretrainGNN(torch.nn.Module):
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

        # their default
        self.ema_updater = EMA(0.99)

        self.online_encoder = Backbone(
            conv,
            hid_dim,
            num_encode_layers,
            num_conv_layers,
            num_mlp_layers,
            backbone_pred_layers,
            norm,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)

        assert num_pred_layers > 0  # IGSD must have a predictor
        self.predictor = MLP([hid_dim] * (num_pred_layers + 1), norm=None)

    def forward(self, data, ppr_data):
        # encoder + projector
        online_embedding, *_ = self.online_encoder(data)
        with torch.no_grad():
            # encoder + projector
            target_embedding, *_ = self.target_encoder(ppr_data)
            # update weights
            update_moving_average(self.ema_updater, self.target_encoder, self.online_encoder)

        online_z = self.predictor(online_embedding)
        target_z = self.predictor(target_embedding)

        online_embedding = F.normalize(online_embedding, dim=-1, p=2)
        target_embedding = F.normalize(target_embedding, dim=-1, p=2)

        consistence_mat = torch.norm(online_z[:, None, :] - target_embedding[None, :, :], dim=-1, p=2) + \
                          torch.norm(target_z[:, None, :] - online_embedding[None, :, :], dim=-1, p=2)

        return consistence_mat
