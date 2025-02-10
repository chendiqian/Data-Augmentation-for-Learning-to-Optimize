import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PlainGNNTrainer:
    def __init__(self, loss_type):
        self.best_objgap = 1.e8
        self.patience = 0
        self.loss_type = loss_type
        if loss_type == 'l2':
            self.loss_func = lambda x: x ** 2
        elif loss_type == 'l1':
            self.loss_func = lambda x: x.abs()
        else:
            raise ValueError

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(device)

            obj_pred = model(data)
            label = data.obj_solution

            loss = self.loss_func(obj_pred - label).mean()

            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            # use both L2 loss and Cos similarity loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses.item() / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        objgaps = []
        for i, data in enumerate(dataloader):
            data = data.to(device)
            opt_obj = data.obj_solution
            obj_pred = model(data)
            obj_gap = torch.abs((opt_obj - obj_pred) / opt_obj)
            objgaps.append(obj_gap)

        objgaps = torch.cat(objgaps, dim=0).mean().item()
        return objgaps


class NTXentPretrainer:
    """
    For graph level contrastive, where we have 2N views of the graph batch
    """
    def __init__(self, temperature):
        self.best_val_loss = 1.e8
        self.best_val_acc = 0.
        self.patience = 0
        self.loss_func = NTXentLoss(temperature=temperature)

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        corrects = 0
        for i, (data1, data2) in enumerate(dataloader):
            optimizer.zero_grad()
            data1 = data1.to(device)
            data2 = data2.to(device)

            pred1 = model(data1)
            pred2 = model(data2)

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

            # top 5 acc
            cos = F.cosine_similarity(pred1.detach()[:, None], pred2.detach()[None, :], dim=-1)
            pos_mask = torch.eye(cos.shape[0], device=device).bool()
            comb_sim = torch.cat([cos[pos_mask][:, None], cos.masked_fill_(pos_mask, -1e10)], dim=1)
            sim_argsort = comb_sim.argsort(dim=1, descending=True).argmin(dim=-1)
            corrects += (sim_argsort < 5).sum()

        return train_losses.item() / num_graphs, corrects.item() / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        val_losses = 0.
        num_graphs = 0
        corrects = 0
        for i, (data1, data2) in enumerate(dataloader):
            data1 = data1.to(device)
            data2 = data2.to(device)

            pred1 = model(data1)
            pred2 = model(data2)

            pred = torch.cat([pred1, pred2], dim=0)
            label = torch.arange(pred1.shape[0], device=device).repeat(2)

            # Basically you need to create labels such that positive pairs share the same label.
            # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/179
            loss = self.loss_func(pred, label)

            val_losses += loss.detach() * data1.num_graphs
            num_graphs += data1.num_graphs

            # top 5 acc
            cos = F.cosine_similarity(pred1[:, None], pred2[None, :], dim=-1)
            pos_mask = torch.eye(cos.shape[0], device=device).bool()
            comb_sim = torch.cat([cos[pos_mask][:, None], cos.masked_fill_(pos_mask, -1e10)], dim=1)
            sim_argsort = comb_sim.argsort(dim=1, descending=True).argmin(dim=-1)
            corrects += (sim_argsort < 5).sum()

        return val_losses.item() / num_graphs, corrects.item() / num_graphs


class LinearTrainer:
    def __init__(self, loss_type):
        self.best_objgap = 1.e8
        self.patience = 0
        self.loss_type = loss_type
        if loss_type == 'l2':
            self.loss_func = lambda x: x ** 2
        elif loss_type == 'l1':
            self.loss_func = lambda x: x.abs()
        else:
            raise ValueError

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, (data, label) in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(device)

            obj_pred = model(data).squeeze()
            loss = self.loss_func(obj_pred - label).mean()
            train_losses += loss.detach() * data.shape[0]
            num_graphs += data.shape[0]

            # use both L2 loss and Cos similarity loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses.item() / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        objgaps = []
        for i, (data, label) in enumerate(dataloader):
            data = data.to(device)
            obj_pred = model(data).squeeze()
            obj_gap = torch.abs((label - obj_pred) / label)
            objgaps.append(obj_gap)

        objgaps = torch.cat(objgaps, dim=0).mean().item()
        return objgaps
