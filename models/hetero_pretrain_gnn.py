import torch
from torch_geometric.nn import MLP, global_mean_pool, global_max_pool

from models.hetero_encoder import BipartiteHeteroEncoder, TripartiteHeteroEncoder


class BipartiteHeteroPretrainGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 head,
                 concat,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 hid_pred,
                 num_mlp_layers,
                 norm,
                 pooling):
        super().__init__()

        self.encoder = BipartiteHeteroEncoder(
            conv,
            head,
            concat,
            hid_dim,
            num_encode_layers,
            num_conv_layers,
            num_mlp_layers,
            norm)

        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        # elif pooling is None:
        #     self.pool = lambda x, *args: x
        else:
            raise NotImplementedError

        if hid_pred == -1:
            hid_pred = hid_dim
        self.predictor = MLP([hid_dim * 2] + [hid_pred] * num_pred_layers, norm=None)

    def forward(self, data):
        x_dict = self.encoder(data)
        embedding = torch.cat([self.pool(x_dict['vals'], data['vals'].batch),
                               self.pool(x_dict['cons'], data['cons'].batch)], dim=1)
        embedding = self.predictor(embedding)
        return embedding


class TripartiteHeteroPretrainGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 head,
                 concat,
                 hid_dim,
                 num_encode_layers,
                 num_conv_layers,
                 num_pred_layers,
                 hid_pred,
                 num_mlp_layers,
                 norm,
                 pooling):
        super().__init__()

        self.encoder = TripartiteHeteroEncoder(
            conv,
            head,
            concat,
            hid_dim,
            num_encode_layers,
            num_conv_layers,
            num_mlp_layers,
            norm
        )

        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        # elif pooling is None:
        #     self.pool = lambda x, *args: x
        # todo: use the obj node
        else:
            raise NotImplementedError

        if hid_pred == -1:
            hid_pred = hid_dim
        self.predictor = MLP([hid_dim * 2] + [hid_pred] * num_pred_layers, norm=None)

    def forward(self, data):
        x_dict = self.encoder(data)
        embedding = torch.cat([self.pool(x_dict['vals'], data['vals'].batch),
                               self.pool(x_dict['cons'], data['cons'].batch)], dim=1)
        embedding = self.predictor(embedding)
        return embedding
