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
            norm
        )

        self.predictor = MLP([hid_dim] * num_pred_layers + [1], norm=None)

    def forward(self, data):
        obj_pred = self.encoder(data)
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
            norm
        )

        if num_pred_layers == 0:
            self.predictor = torch.nn.Identity()
        else:
            self.predictor = MLP([hid_dim] * (num_pred_layers + 1), norm=None)

    def forward(self, data):
        obj_embedding = self.encoder(data)
        embedding = self.predictor(obj_embedding)
        return embedding
