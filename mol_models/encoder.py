import torch
from torch_geometric.nn import MLP, global_mean_pool

from mol_models.convs.gcnconv import GCNConv
from mol_models.convs.ginconv import GINEConv


def get_conv_layer(conv: str,
                   hid_dim: int,
                   num_mlp_layers: int,
                   norm: str):
    if conv.lower() == 'gcnconv':
        return GCNConv(edge_dim=hid_dim,
                       hid_dim=hid_dim,
                       num_mlp_layers=num_mlp_layers,
                       norm=norm)
    elif conv.lower() == 'ginconv':
        return GINEConv(edge_dim=hid_dim,
                        hid_dim=hid_dim,
                        num_mlp_layers=num_mlp_layers,
                        norm=norm)
    else:
        raise NotImplementedError


class ZINCBondEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(ZINCBondEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, edge_attr):
        if edge_attr is not None:
            return self.embedding(edge_attr)
        else:
            return None


class ZINCAtomEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(ZINCAtomEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=28, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x.squeeze(1)
        masked = x == -1
        x[masked] = 0  # temporary
        x = self.embedding(x)
        x[masked] = x[masked] * 0.
        return x


class Encoder(torch.nn.Module):
    def __init__(self,
                 conv,
                 hid_dim,
                 num_conv_layers,
                 num_mlp_layers,
                 num_backbone_mlp,
                 norm):
        super().__init__()

        self.num_layers = num_conv_layers
        self.atom_encoder = ZINCAtomEncoder(hid_dim)
        self.bond_encoder = ZINCBondEncoder(hid_dim)

        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.gcns.append(get_conv_layer(conv, hid_dim, num_mlp_layers, norm))

        if num_backbone_mlp > 0:
            self.fc_obj = MLP([hid_dim] * (num_backbone_mlp + 1), norm=None)
        else:
            self.fc_obj = torch.nn.Identity()

    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)

        for i in range(self.num_layers):
            x = self.gcns[i](x, data.edge_index, edge_attr, data.batch, data.norm if hasattr(data, 'norm') else None)

        graph_emb = global_mean_pool(x, data.batch)
        pred = self.fc_obj(graph_emb)
        return pred
