from transforms.graph_cl import GraphCLDropNode, GraphCLMaskNode, GraphCLPerturbEdge
from transforms.lp_preserve import (DropInactiveConstraint,
                                    OracleDropInactiveConstraint,
                                    AddRedundantConstraint,
                                    ScaleConstraint, ScaleObj,
                                    AddOrthogonalConstraint,
                                    AddDumbVariables)
from transforms.rw_subgraph import RWSubgraph
from transforms.ppr_augment import PageRankAugment


# priority, the smaller, the higher
TRANSFORM_CODEBOOK = {
    # GraphCL:
    GraphCLDropNode: 1,
    GraphCLMaskNode: 2,
    GraphCLPerturbEdge: 0,

    # GCC:
    RWSubgraph: 0,

    # IGSD:
    PageRankAugment: 0,

    # ours
    OracleDropInactiveConstraint: 100,
    DropInactiveConstraint: 100,   # todo: experimental
    AddDumbVariables: 4,
    AddRedundantConstraint: 2,
    ScaleConstraint: 1,
    ScaleObj: 1,
    AddOrthogonalConstraint: 3,  # this after redundant cons, as this might introduce active constraints
}

__all__ = [
    'GraphCLDropNode',
    'GraphCLMaskNode',
    'GraphCLPerturbEdge',

    'RWSubgraph',

    'PageRankAugment',

    'OracleDropInactiveConstraint',
    'DropInactiveConstraint',
    'AddRedundantConstraint',
    'AddDumbVariables',
    'AddOrthogonalConstraint',
    'ScaleConstraint',
    'ScaleObj',
]