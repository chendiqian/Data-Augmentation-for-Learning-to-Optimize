import torch
from torch_geometric.nn import MLP, Linear

from models.hetero_backbone import BipartiteHeteroBackbone


class BipartiteHeteroGNN(torch.nn.Module):
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

        self.encoder = BipartiteHeteroBackbone(
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


class BipartiteHeteroPretrainGNN(torch.nn.Module):
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

        self.encoder = BipartiteHeteroBackbone(
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

        self.encoder = BipartiteHeteroBackbone(
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
        obj_embedding, node_embedding = self.encoder(data)
        obj_embedding = self.predictor(obj_embedding)
        node_embedding = self.fc_nodes(node_embedding)
        node_embedding = self.node_predictor(node_embedding)
        return obj_embedding, node_embedding
