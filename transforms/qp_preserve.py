import math
from typing import Tuple

import numpy as np
import torch
from scipy import sparse as sp
from torch_geometric.data import HeteroData
from torch_geometric.utils import bipartite_subgraph
from torch_scatter import scatter_sum


class QPScaleCoordinate:
    """
    Imagine the coordinates are expanded, with c also expanded correspondingly,
    then the optimal solution is scaled, but the objective is unchanged
    """

    def __init__(self, strength=1.):
        assert strength > 0
        # we scale all the constraints, but with variable strength
        self.p = strength

    def __call__(self, data: HeteroData) -> HeteroData:
        m, n = data['cons'].num_nodes, data['vals'].num_nodes

        # scales = abs(1 + N(0, 1) * exp(p - 1))
        noise_scale = math.exp(self.p) - 1.
        # 0. -> 0., 1. -> 1.73

        noise = torch.randn(n) * noise_scale
        scales = (noise + torch.ones_like(noise)).abs()

        Acols = data[('cons', 'to', 'vals')].edge_index[1]
        Qrows, Qcols = data[('vals', 'to', 'vals')].edge_index

        new_data = data.__class__(
            cons={
                'num_nodes': m,
                'x': data['cons'].x,
            },
            vals={
                'num_nodes': n,
                'x': data['vals'].x,
            },
            cons__to__vals={'edge_index': data[('cons', 'to', 'vals')].edge_index,
                            'edge_attr': data[('cons', 'to', 'vals')].edge_attr * scales[Acols, None]},
            vals__to__vals={'edge_index': data[('vals', 'to', 'vals')].edge_index,
                            'edge_attr': data[('vals', 'to', 'vals')].edge_attr * scales[Qcols, None] * scales[Qrows, None]},
            q=data.q * scales,
            b=data.b,
            obj_solution=data.obj_solution,
            x_solution=data.x_solution / scales,
            inactive_idx=data.inactive_idx,
        )
        return new_data

    def __repr__(self):
        return 'QPScaleCoordinate'
