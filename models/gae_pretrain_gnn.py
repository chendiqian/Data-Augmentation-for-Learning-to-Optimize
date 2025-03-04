import torch
from torch_geometric.nn import MLP

from models.backbone import Backbone


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
