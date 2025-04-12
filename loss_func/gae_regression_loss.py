import torch
from torch_scatter import scatter_mean


def gae_regression_loss(vals_embedding, cons_embedding,
                        c2v_edge_index, c2v_edge_attr,
                        v2v_edge_index, v2v_edge_attr,
                        c2v_edge_slice, v2v_edge_slice,
                        num_graphs):
    c2v_pred_edges = (vals_embedding[c2v_edge_index[1]] * cons_embedding[c2v_edge_index[0]]).sum(1)
    c2v_loss = torch.nn.MSELoss(reduction='none')(c2v_pred_edges, c2v_edge_attr)
    c2v_nedges = c2v_edge_slice[1:] - c2v_edge_slice[:-1]
    c2v_batch = torch.arange(num_graphs, device=vals_embedding.device).repeat_interleave(c2v_nedges)
    loss = scatter_mean(c2v_loss, c2v_batch, dim=0).mean()

    if v2v_edge_index is not None:
        # reconstruction for QP
        v2v_pred_edges = (vals_embedding[v2v_edge_index[1]] * vals_embedding[v2v_edge_index[0]]).sum(1)
        v2v_loss = torch.nn.MSELoss(reduction='none')(v2v_pred_edges, v2v_edge_attr)
        v2v_nedges = v2v_edge_slice[1:] - v2v_edge_slice[:-1]
        v2v_batch = torch.arange(num_graphs, device=vals_embedding.device).repeat_interleave(v2v_nedges)
        v2v_loss = scatter_mean(v2v_loss, v2v_batch, dim=0).mean()
        loss = loss + v2v_loss
    return loss
