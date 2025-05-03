import torch
from pytorch_metric_learning.losses import NTXentLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NTXentPretrainer:
    """
    For graph level contrastive, where we have 2N views of the graph batch
    """
    def __init__(self, temperature):
        self.best_loss = 1.e8
        self.best_acc = 0.
        self.patience = 0
        self.loss_func = NTXentLoss(temperature=temperature)

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, (data1, data2) in enumerate(dataloader):
            optimizer.zero_grad()
            data1 = data1.to(device)
            data2 = data2.to(device)

            pred1, _ = model(data1)
            pred2, _ = model(data2)

            pred = torch.cat([pred1, pred2], dim=0)
            label = torch.arange(pred1.shape[0], device=device).repeat(2)

            # Basically you need to create labels such that positive pairs share the same label.
            # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/179
            loss = self.loss_func(pred, label)

            train_losses += loss.detach() * data1.num_graphs
            num_graphs += data1.num_graphs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses.item() / num_graphs
