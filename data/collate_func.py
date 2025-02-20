from typing import List, Tuple
import itertools

import torch
from torch_geometric.data import Data, Batch


def collate_fn_lp_base(graphs: List[Data]):
    new_batch = Batch.from_data_list(graphs, exclude_keys=['x', 'nulls'])  # we drop the dumb x features
    # finish the half of symmetric edges
    flip_tensor = torch.tensor([1, 0])
    for k, v in graphs[0].edge_index_dict.items():
        src, rel, dst = k
        new_batch[(dst, rel, src)].edge_index = new_batch[(src, rel, dst)].edge_index[flip_tensor]
        new_batch[(dst, rel, src)].edge_attr = new_batch[(src, rel, dst)].edge_attr

    if not hasattr(new_batch[('vals', 'to', 'cons')], 'norm'):
        norm_dict = {}
        for k, v in new_batch.edge_index_dict.items():
            norm_dict[k] = None
        new_batch.norm_dict = norm_dict

    return new_batch


def collate_pos_pair(graphs: List[Tuple[Data, Data]]):
    graphs1, graphs2 = zip(*graphs)
    batch1 = collate_fn_lp_base(graphs1)
    batch2 = collate_fn_lp_base(graphs2)
    return batch1, batch2


def collate_pos_neg_pair(lists_graphs: List[List[Data]]):
    data = [ls.pop(0) for ls in lists_graphs]
    pos = [ls.pop(0) for ls in lists_graphs]
    negs = list(itertools.chain(*lists_graphs))
    return collate_fn_lp_base(data), collate_fn_lp_base(pos), collate_fn_lp_base(negs)
