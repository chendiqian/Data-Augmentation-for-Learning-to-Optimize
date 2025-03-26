import torch
from loss_func.mvgrl_l2g_loss import DualBranchContrast
from loss_func.pygcl_utils import JSD

from utils.evaluation import compute_acc

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MVGRLPretrainer:
    def __init__(self):
        self.best_loss = 1.e8
        self.best_acc = 0.
        self.patience = 0
        self.loss_func = DualBranchContrast(JSD(), 'G2L')

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        corrects = 0
        for i, (anchor, aug) in enumerate(dataloader):
            optimizer.zero_grad()
            anchor = anchor.to(device)
            aug = aug.to(device)

            n1, n2, g1, g2 = model(anchor, aug)
            vals_nnodes = (anchor['vals'].ptr[1:] - anchor['vals'].ptr[:-1]).to(device)
            cons_nnodes = (anchor['cons'].ptr[1:] - anchor['cons'].ptr[:-1]).to(device)
            batch = torch.cat([torch.arange(anchor.num_graphs, device=device).repeat_interleave(vals_nnodes),
                               torch.arange(anchor.num_graphs, device=device).repeat_interleave(cons_nnodes)], dim=0)
            loss = self.loss_func(n1, n2, g1, g2, batch)

            train_losses += loss.detach() * anchor.num_graphs
            num_graphs += anchor.num_graphs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

            corrects += compute_acc(g1, g2)

        return train_losses.item() / num_graphs, corrects.item() / num_graphs
