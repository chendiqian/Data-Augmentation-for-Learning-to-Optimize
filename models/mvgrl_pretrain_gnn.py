import copy

import torch
from torch_geometric.nn import MLP

from models.backbone import Backbone


class MVGRLPretrainGNN(torch.nn.Module):
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

        self.encoder1 = Backbone(
            conv,
            hid_dim,
            num_encode_layers,
            num_conv_layers,
            num_mlp_layers,
            backbone_pred_layers,
            norm,
            output_nodes=True,
        )

        # copy, so that to keep the computation graph
        self.encoder2 = copy.deepcopy(self.encoder1)
        # in place
        self.encoder2.reset_parameters()
        # according to the paper, the GNN is not shared, but the MLP is shared
        self.encoder2.fc_obj = self.encoder1.fc_obj

        # to make the backbone consistent, we move the node FF here
        # also it is shared according the paper
        if backbone_pred_layers == 0:
            self.fc_nodes = torch.nn.Identity()
        else:
            self.fc_nodes = MLP([hid_dim] * (backbone_pred_layers + 1), norm=None)

        assert num_pred_layers > 0  # must have a predictor
        self.predictor = MLP([hid_dim] * (num_pred_layers + 1), norm=None)
        self.node_predictor = MLP([hid_dim] * (num_pred_layers + 1), norm=None)

    def forward(self, anchor_data, aug_data):
        obj_embedding1, x_dict1 = self.encoder1(anchor_data)
        node_embedding1 = torch.cat([x_dict1['vals'], x_dict1['cons']], dim=0)
        obj_embedding1 = self.predictor(obj_embedding1)
        node_embedding1 = self.fc_nodes(node_embedding1)
        node_embedding1 = self.node_predictor(node_embedding1)

        obj_embedding2, x_dict2 = self.encoder2(aug_data)
        node_embedding2 = torch.cat([x_dict2['vals'], x_dict2['cons']], dim=0)
        obj_embedding2 = self.predictor(obj_embedding2)
        node_embedding2 = self.fc_nodes(node_embedding2)
        node_embedding2 = self.node_predictor(node_embedding2)
        return node_embedding1, node_embedding2, obj_embedding1, obj_embedding2
