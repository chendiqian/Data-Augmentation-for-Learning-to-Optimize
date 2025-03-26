import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class IGSDPretrainer:
    def __init__(self, temperature):
        self.best_loss = 1.e8
        self.best_acc = 0.
        self.patience = 0
        self.temperature = temperature
        self.loss_func = torch.nn.CrossEntropyLoss()

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        corrects = 0
        for i, (data1, data2) in enumerate(dataloader):
            optimizer.zero_grad()
            data1 = data1.to(device)
            data2 = data2.to(device)

            consistence_mat = model(data1, data2)
            labels = torch.arange(consistence_mat.shape[0], device=device)
            # it is the L2 norm of distance, so the distance should be the lower the better
            loss = self.loss_func(-consistence_mat / self.temperature, labels)

            train_losses += loss.detach() * data1.num_graphs
            num_graphs += data1.num_graphs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

            preds = consistence_mat.argmin(1)
            corrects += (preds == labels).sum()

        return train_losses.item() / num_graphs, corrects.item() / num_graphs
