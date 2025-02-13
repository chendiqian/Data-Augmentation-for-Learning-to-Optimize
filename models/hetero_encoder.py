from typing import Dict, Optional

import torch
from torch_geometric.nn import MLP
from torch_geometric.typing import EdgeType, NodeType

from models.hetero_conv import TripartiteConv
from models.convs.gcn2conv import GCN2Conv
from models.convs.gcnconv import GCNConv
from models.convs.genconv import GENConv
from models.convs.ginconv import GINEConv
from models.nn_utils import LogEncoder


def get_conv_layer(conv: str,
                   hid_dim: int,
                   num_mlp_layers: int,
                   norm: str):
    if conv.lower() == 'genconv':
        return GENConv(in_channels=-1,
                       out_channels=hid_dim,
                       num_layers=num_mlp_layers,
                       aggr='softmax',
                       msg_norm=norm is not None,
                       learn_msg_scale=norm is not None,
                       norm=norm,
                       bias=True,
                       edge_dim=1)
    elif conv.lower() == 'gcnconv':
        return GCNConv(edge_dim=1,
                       hid_dim=hid_dim,
                       num_mlp_layers=num_mlp_layers,
                       norm=norm)
    elif conv.lower() == 'ginconv':
        return GINEConv(edge_dim=1,
                        hid_dim=hid_dim,
                        num_mlp_layers=num_mlp_layers,
                        norm=norm)
    elif conv.lower() == 'gcn2conv':
        return GCN2Conv(edge_dim=1,
                        hid_dim=hid_dim,
                        num_mlp_layers=num_mlp_layers,
                        norm=norm)
    else:
        raise NotImplementedError


class TripartiteHeteroEncoder(torch.nn.Module):
    def __init__(self,
                 conv,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_mlp_layers,
                 num_pred_layers,
                 norm):
        super().__init__()

        self.num_layers = num_conv_layers
        self.b_encoder = MLP([1] + [hid_dim] * num_encode_layers, norm=None)
        self.q_encoder = MLP([1] + [hid_dim] * num_encode_layers, norm=None)

        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gcns.append(TripartiteConv(
                v2c_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm),
                c2v_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm),
                # 1 node only so no normalization
                v2o_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, None),
                o2v_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm),
                c2o_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, None),
                o2c_conv=get_conv_layer(conv, hid_dim, num_mlp_layers, norm),
            ))

        # self.fc_cons = MLP([hid_dim] * num_pred_layers, norm=None)
        # self.fc_vals = MLP([hid_dim] * num_pred_layers, norm=None)
        self.fc_obj = MLP([hid_dim] * num_pred_layers, norm=None)

    def init_embedding(self, data):
        batch_dict: Dict[NodeType, torch.LongTensor] = data.batch_dict
        edge_index_dict: Dict[EdgeType, torch.LongTensor] = data.edge_index_dict
        edge_attr_dict: Dict[EdgeType, torch.FloatTensor] = data.edge_attr_dict
        norm_dict: Dict[EdgeType, Optional[torch.FloatTensor]] = data.norm_dict

        cons_embedding = self.b_encoder(data.b[:, None])
        vals_embedding = self.q_encoder(data.q[:, None])

        x_dict: Dict[NodeType, torch.FloatTensor] = {
            'vals': vals_embedding,
            'cons': cons_embedding,
            # dumb initialization
            'obj': vals_embedding.new_zeros(data['obj'].num_nodes, vals_embedding.shape[1])}
        x0_dict: Dict[NodeType, torch.FloatTensor] = {'vals': vals_embedding,
                                                      'cons': cons_embedding,
                                                      'obj': x_dict['obj']}
        return batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict, x0_dict

    def forward(self, data):
        batch_dict, edge_index_dict, edge_attr_dict, norm_dict, x_dict, x0_dict = self.init_embedding(data)

        for i in range(self.num_layers):
            x_dict = self.gcns[i](x_dict, x0_dict, batch_dict, edge_index_dict, edge_attr_dict, norm_dict)

        # x_dict['vals'] = self.fc_vals(x_dict['vals'])
        # x_dict['cons'] = self.fc_cons(x_dict['cons'])
        pred = self.fc_obj(x_dict['obj'])
        return pred
