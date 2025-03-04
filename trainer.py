import torch
from pytorch_metric_learning.losses import NTXentLoss

from data.n_pair_loss import NPairLoss
from data.utils import compute_acc
from data.infograph_loss import local_global_loss

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

            corrects += compute_acc(pred1, pred2)

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

            pred1, _ = model(data1)
            pred2, _ = model(data2)

            pred = torch.cat([pred1, pred2], dim=0)
            label = torch.arange(pred1.shape[0], device=device).repeat(2)

            # Basically you need to create labels such that positive pairs share the same label.
            # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/179
            loss = self.loss_func(pred, label)

            val_losses += loss.detach() * data1.num_graphs
            num_graphs += data1.num_graphs

            corrects += compute_acc(pred1, pred2)

        return val_losses.item() / num_graphs, corrects.item() / num_graphs


class NPairPretrainer:
    """
    For graph level contrastive, where we have anchor, 1 pos and N neg samples
    """
    def __init__(self, temperature):
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


class InfoGraphPretrainer:
    def __init__(self):
        self.best_val_loss = 1.e8
        self.best_val_acc = 0.
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

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        val_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)

            global_pred, node_pred = model(data)
            # homogeneous number of nodes
            vals_nnodes = (data['vals'].ptr[1:] - data['vals'].ptr[:-1]).to(device)
            cons_nnodes = (data['cons'].ptr[1:] - data['cons'].ptr[:-1]).to(device)
            loss = local_global_loss(node_pred, global_pred, vals_nnodes, cons_nnodes)

            val_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

        return val_losses.item() / num_graphs, 0.


class IGSDPretrainer:
    def __init__(self, temperature):
        self.best_val_loss = 1.e8
        self.best_val_acc = 0.
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

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        val_losses = 0.
        num_graphs = 0
        corrects = 0
        for i, (data1, data2) in enumerate(dataloader):
            data1 = data1.to(device)
            data2 = data2.to(device)

            consistence_mat = model(data1, data2)
            labels = torch.arange(consistence_mat.shape[0], device=device)
            # it is the L2 norm of distance, so the distance should be the lower the better
            loss = self.loss_func(-consistence_mat / self.temperature, labels)

            val_losses += loss.detach() * data1.num_graphs
            num_graphs += data1.num_graphs

            preds = consistence_mat.argmin(1)
            corrects += (preds == labels).sum()

        return val_losses.item() / num_graphs, corrects.item() / num_graphs
