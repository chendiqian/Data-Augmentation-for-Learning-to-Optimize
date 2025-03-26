import torch

from loss_func.infograph_loss import local_global_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class InfoGraphPretrainer:
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

            global_pred, node_pred = model(data)
            # homogeneous number of nodes
            vals_nnodes = (data['vals'].ptr[1:] - data['vals'].ptr[:-1]).to(device)
            cons_nnodes = (data['cons'].ptr[1:] - data['cons'].ptr[:-1]).to(device)
            loss = local_global_loss(node_pred, global_pred, vals_nnodes, cons_nnodes)

            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses.item() / num_graphs, 0.
