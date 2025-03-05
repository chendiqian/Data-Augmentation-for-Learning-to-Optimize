import torch
from loss_func.dgi_loss import SingleBranchContrast
from loss_func.pygcl_utils import JSD

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DGIPretrainer:
    def __init__(self):
        self.best_val_loss = 1.e8
        self.best_val_acc = 0.
        self.patience = 0
        self.loss_func = SingleBranchContrast(JSD(), 'G2L')

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()

            obj_embedding, node_embedding = model(data)
            vals_nnodes = (data['vals'].ptr[1:] - data['vals'].ptr[:-1]).to(device)
            cons_nnodes = (data['cons'].ptr[1:] - data['cons'].ptr[:-1]).to(device)
            batch = torch.cat([torch.arange(data.num_graphs, device=device).repeat_interleave(vals_nnodes),
                               torch.arange(data.num_graphs, device=device).repeat_interleave(cons_nnodes)], dim=0)
            loss = self.loss_func(node_embedding, obj_embedding, batch)

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
            obj_embedding, node_embedding = model(data)
            vals_nnodes = (data['vals'].ptr[1:] - data['vals'].ptr[:-1]).to(device)
            cons_nnodes = (data['cons'].ptr[1:] - data['cons'].ptr[:-1]).to(device)
            batch = torch.cat([torch.arange(data.num_graphs, device=device).repeat_interleave(vals_nnodes),
                               torch.arange(data.num_graphs, device=device).repeat_interleave(cons_nnodes)], dim=0)
            loss = self.loss_func(node_embedding, obj_embedding, batch)

            val_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

        return val_losses.item() / num_graphs, 0.
