import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PlainGNNTrainer:
    def __init__(self):
        self.best_objgap = 1.e8
        self.patience = 0
        self.loss_func = torch.nn.MSELoss(reduction='mean')

    def train_step(self, data, label, model, optimizer):
        optimizer.zero_grad()
        obj_pred = model(data)
        loss = self.loss_func(obj_pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach()

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)
            label = data.obj_solution
            loss = self.train_step(data, label, model, optimizer)
            train_losses += loss * data.num_graphs
            num_graphs += data.num_graphs

        return train_losses.item() / num_graphs

    def val_step(self, data, obj, model):
        obj_pred = model(data)
        obj_gap = torch.abs((obj - obj_pred) / obj)
        return obj_gap

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        objgaps = []
        for i, data in enumerate(dataloader):
            data = data.to(device)
            obj_gap = self.val_step(data, data.obj_solution, model)
            objgaps.append(obj_gap)

        objgaps = torch.cat(objgaps, dim=0).mean().item()
        return objgaps


class QPLIBTrainer:
    def __init__(self):
        self.best_objgap = 1.e8
        self.patience = 0
        self.loss_func = torch.nn.MSELoss(reduction='mean')

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)
            label = data.obj_solution
            optimizer.zero_grad()
            obj_pred = model(data)
            loss = self.loss_func(obj_pred, label)

            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses.item() / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        losses = 0.
        for i, data in enumerate(dataloader):
            data = data.to(device)
            obj_pred = model(data)
            loss = self.loss_func(obj_pred, data.obj_solution)
            losses += loss

        return losses.item() / len(dataloader)


class LinearTrainer(PlainGNNTrainer):
    """
    The only difference is this trainer uses a loader that gives us the labels
    """

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, (data, label) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)
            loss = self.train_step(data, label, model, optimizer)
            train_losses += loss * label.shape[0]
            num_graphs += label.shape[0]
            train_losses += loss * data.shape[0]
            num_graphs += data.shape[0]

        return train_losses.item() / num_graphs

    @torch.no_grad()
    def eval(self, dataloader, model):
        model.eval()

        objgaps = []
        for i, (data, label) in enumerate(dataloader):
            data = data.to(device)
            obj_gap = self.val_step(data, label, model)
            objgaps.append(obj_gap)

        objgaps = torch.cat(objgaps, dim=0).mean().item()
        return objgaps
