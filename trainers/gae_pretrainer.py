import torch
from loss_func.gae_regression_loss import gae_regression_loss

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

            loss = gae_regression_loss(vals, cons,
                                       data[('cons', 'to', 'vals')].edge_index,
                                       data[('cons', 'to', 'vals')].edge_attr.squeeze(1),
                                       data._slice_dict[('cons', 'to', 'vals')]['edge_index'].to(device),
                                       data.num_graphs)
            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses.item() / num_graphs, 0.
