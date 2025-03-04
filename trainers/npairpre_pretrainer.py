import torch

from loss_func.n_pair_loss import NPairLoss


class NPairPretrainer:
    """
    For graph level contrastive, where we have anchor, 1 pos and N neg samples
    """
    def __init__(self, temperature):
        raise DeprecationWarning

        self.best_val_loss = 1.e8
        self.best_val_acc = 0.
        self.patience = 0
        self.loss_func = NPairLoss(temperature=temperature)

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        corrects = 0
        for i, (data, pos, negs) in enumerate(dataloader):
            optimizer.zero_grad()

            anchor = model(data)
            pos = model(pos)
            negs = model(negs)  # a1, a2, a3, a4, b1, b2, b3, b4
            batchsize, features = anchor.shape
            negs = negs.reshape(batchsize, -1, features)

            loss, logits = self.loss_func(anchor, pos, negs)

            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

            # acc
            preds = logits.detach().argmax(dim=1)
            corrects += (preds == 0).sum()

        return train_losses.item() / num_graphs, corrects.item() / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        val_losses = 0.
        num_graphs = 0
        corrects = 0
        for i, (data, pos, negs) in enumerate(dataloader):
            anchor = model(data)
            pos = model(pos)
            negs = model(negs)  # a1, a2, a3, a4, b1, b2, b3, b4
            batchsize, features = anchor.shape
            negs = negs.reshape(batchsize, -1, features)

            loss, logits = self.loss_func(anchor, pos, negs)

            val_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            # acc
            preds = logits.detach().argmax(dim=1)
            corrects += (preds == 0).sum()

        return val_losses.item() / num_graphs, corrects.item() / num_graphs
