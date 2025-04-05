from typing import Dict, Optional

import torch
from torch_geometric.nn import MLP, global_mean_pool
from torch_geometric.typing import EdgeType, NodeType

from models.hetero_conv import LPHeteroConv, QPHeteroConv
from models.convs.gcnconv import GCNConv
from models.convs.ginconv import GINEConv
from models.convs.sageconv import SAGEConv


def get_conv_layer(conv: str,
                   hid_dim: int,
                   num_mlp_layers: int,
                   norm: str):
    if conv.lower() == 'gcnconv':
        return GCNConv(hid_dim=hid_dim,
                       num_mlp_layers=num_mlp_layers,
                       norm=norm)
    elif conv.lower() == 'ginconv':
        return GINEConv(hid_dim=hid_dim,
                        num_mlp_layers=num_mlp_layers,
                        norm=norm)
    elif conv.lower() == 'sageconv':
        return SAGEConv(hid_dim=hid_dim, num_mlp_layers=num_mlp_layers, norm=norm)
    else:
        raise NotImplementedError


class Backbone(torch.nn.Module):
    def __init__(self,
                 is_qp,
                 conv,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_mlp_layers,
                 backbone_pred_layers,
                 norm,
                 output_nodes=False):
        super().__init__()
        self.output_nodes = output_nodes
        self.num_layers = num_conv_layers
        self.b_encoder = MLP([1] + [hid_dim] * num_encode_layers, norm=None)
        self.q_encoder = MLP([1] + [hid_dim] * num_encode_layers, norm=None)

        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            if is_qp:
                self.gcns.append(QPHeteroConv(
                    v2v_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm),
                    v2c_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm),
                    c2v_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm)
                ))
            else:
                self.gcns.append(LPHeteroConv(
                    v2c_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm),
                    c2v_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm)
                ))

        if backbone_pred_layers == 0:
            self.fc_obj = torch.nn.Identity()
        else:
            self.fc_obj = MLP([hid_dim] * (backbone_pred_layers + 1), norm=None)

    def reset_parameters(self):
        self.b_encoder.reset_parameters()
        self.q_encoder.reset_parameters()
        for g in self.gcns:
            g.reset_parameters()
        self.fc_obj.reset_parameters()

    def init_embedding(self, data):
        batch_dict: Dict[NodeType, torch.LongTensor] = data.batch_dict
        edge_index_dict: Dict[EdgeType, torch.LongTensor] = data.edge_index_dict
        edge_attr_dict: Dict[EdgeType, torch.FloatTensor] = data.edge_attr_dict
        norm_dict: Dict[EdgeType, Optional[torch.FloatTensor]] = data.norm_dict

        cons_embedding = self.b_encoder(data.b[:, None])
        vals_embedding = self.q_encoder(data.q[:, None])

        x_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding, 'cons': cons_embedding}
        return batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict

    def forward(self, data):
        batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict = self.init_embedding(data)

        for i in range(self.num_layers):
            x_dict = self.gcns[i](x_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        global_pred = (global_mean_pool(x_dict['vals'], batch_dict['vals'], data.num_graphs) +
                       global_mean_pool(x_dict['cons'], batch_dict['cons'], data.num_graphs))
        global_pred = self.fc_obj(global_pred)

        if self.output_nodes:
            node_embeddings = x_dict
        else:
            node_embeddings = None
        return global_pred, node_embeddings
