import torch
from loss_func.gae_regression_loss import gae_regression_loss
from utils.evaluation import is_qp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GAEPretrainer:
    def __init__(self):
        self.best_loss = 1.e8
        self.best_acc = 0.
        self.patience = 0

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(device)

            vals, cons = model(data)

            if is_qp(data):
                v2v_edge_index = data[('vals', 'to', 'vals')].edge_index
                v2v_edge_attr = data[('vals', 'to', 'vals')].edge_attr.squeeze(1)
                v2v_slice = data._slice_dict[('vals', 'to', 'vals')]['edge_index'].to(device)
            else:
                v2v_edge_index = v2v_edge_attr = v2v_slice = None

            loss = gae_regression_loss(vals, cons,
                                       data[('cons', 'to', 'vals')].edge_index,
                                       data[('cons', 'to', 'vals')].edge_attr.squeeze(1),
                                       v2v_edge_index, v2v_edge_attr,
                                       data._slice_dict[('cons', 'to', 'vals')]['edge_index'].to(device),
                                       v2v_slice,
                                       data.num_graphs)

            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses.item() / num_graphs, 0.
