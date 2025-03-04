import torch
from torch_geometric.nn import MLP, Linear
import copy
from models.backbone import Backbone
from data.utils import EMA, update_moving_average
import torch.nn.functional as F


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


class PretrainGNN(torch.nn.Module):
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
        )

        if num_pred_layers == 0:
            self.predictor = torch.nn.Identity()
        else:
            self.predictor = MLP([hid_dim] * (num_pred_layers + 1), norm=None)

    def forward(self, data):
        obj_embedding, *_ = self.encoder(data)
        obj_embedding = self.predictor(obj_embedding)
        return obj_embedding, None


class InfoGraphPretrainGNN(torch.nn.Module):
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
            output_nodes=True,
        )

        # to make the backbone consistent, we move the node FF here
        if backbone_pred_layers == 0:
            self.fc_nodes = torch.nn.Identity()
        else:
            self.fc_nodes = MLP([hid_dim] * (backbone_pred_layers + 1), norm=None)

        if num_pred_layers == 0:
            self.predictor = torch.nn.Identity()
            self.node_predictor = torch.nn.Identity()
        else:
            self.predictor = MLP([hid_dim] * (num_pred_layers + 1), norm=None)
            self.node_predictor = MLP([hid_dim] * (num_pred_layers + 1), norm=None)

    def forward(self, data):
        obj_embedding, x_dict = self.encoder(data)
        node_embedding = torch.cat([x_dict['vals'], x_dict['cons']], dim=0)
        obj_embedding = self.predictor(obj_embedding)
        node_embedding = self.fc_nodes(node_embedding)
        node_embedding = self.node_predictor(node_embedding)
        return obj_embedding, node_embedding


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


class GAEPretrainGNN(torch.nn.Module):
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
            output_nodes=True,
        )

        # to make the backbone consistent, we move the node FF here
        if backbone_pred_layers == 0:
            self.fc_nodes = torch.nn.Identity()
        else:
            self.fc_nodes = MLP([hid_dim] * (backbone_pred_layers + 1), norm=None)

        if num_pred_layers == 0:
            self.node_predictor = torch.nn.Identity()
        else:
            self.node_predictor = MLP([hid_dim] * (num_pred_layers + 1), norm=None)

    def forward(self, data):
        _, x_dict = self.encoder(data)
        vals_embedding = self.node_predictor(self.fc_nodes(x_dict['vals']))
        cons_embedding = self.node_predictor(self.fc_nodes(x_dict['cons']))
        return vals_embedding, cons_embedding
