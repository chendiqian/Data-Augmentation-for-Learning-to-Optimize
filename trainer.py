import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_scatter import scatter_sum
from pytorch_metric_learning.losses import NTXentLoss

from data.utils import calc_violation

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PlainGNNTrainer:
    def __init__(self,
                 loss_type,
                 coeff_obj,
                 coeff_vio,
                 ):
        self.best_objgap = 1.e8
        self.patience = 0
        self.loss_type = loss_type
        if loss_type == 'l2':
            self.loss_func = lambda x: x ** 2
        elif loss_type == 'l1':
            self.loss_func = lambda x: x.abs()
        else:
            raise ValueError
        self.coeff_obj = coeff_obj
        self.coeff_vio = coeff_vio

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        train_vios = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(device)

            pred = model(data)
            label = data.x_solution

            loss = self.loss_func(pred - label)
            # mean over each variable in an instance, then mean over instances
            loss = scatter(loss, data['vals'].batch, dim=0, reduce='mean').mean()

            loss_vio = (calc_violation(pred, data) ** 2).mean()

            train_losses += loss.detach() * data.num_graphs
            train_vios += loss_vio.detach() * data.num_graphs
            num_graphs += data.num_graphs

            if self.coeff_vio > 0:
                loss = loss * self.coeff_obj + loss_vio * self.coeff_vio

            # use both L2 loss and Cos similarity loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses.item() / num_graphs, train_vios.item() / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        objgaps = []
        violations = []
        for i, data in enumerate(dataloader):
            data = data.to(device)
            opt_obj = data.obj_solution
            pred_x = model(data).relu()  # (nnodes,)

            batch_obj = scatter_sum(pred_x * data.q, data['vals'].batch, dim=0)
            obj_gap = torch.abs((opt_obj - batch_obj) / opt_obj)
            violation = calc_violation(pred_x, data)

            objgaps.append(obj_gap)
            violations.append(violation)

        objgaps = torch.cat(objgaps, dim=0).mean().item()
        violations = torch.cat(violations, dim=0).mean().item()
        return objgaps, violations


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
