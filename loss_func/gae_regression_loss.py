import torch
from torch_scatter import scatter_mean


def gae_regression_loss(vals_embedding, cons_embedding, c2v_edge_inex, edge_attr, edge_slice, num_graphs):
    pred_edges = (vals_embedding[c2v_edge_inex[1]] * cons_embedding[c2v_edge_inex[0]]).sum(1)
    loss = torch.nn.MSELoss(reduction='none')(pred_edges, edge_attr)
    nedges = edge_slice[1:] - edge_slice[:-1]
    batch = torch.arange(num_graphs, device=vals_embedding.device).repeat_interleave(nedges)
    loss = scatter_mean(loss, batch, dim=0).mean()
    return loss
