from transforms.graph_cl import GraphCLDropNode, GraphCLMaskNode, GraphCLPerturbEdge
from transforms.lp_preserve import (DropInactiveConstraint,
                                    AddRedundantConstraint,
                                    ScaleConstraint, ScaleObj,
                                    AddOrthogonalConstraint,
                                    AddDumbVariables)
from transforms.gcc_subgraph import GCCSubgraph
from transforms.igsd_ppr_augment import IGSDPageRankAugment


# priority, the smaller, the higher
TRANSFORM_CODEBOOK = {
    # GraphCL:
    GraphCLDropNode: 1,
    GraphCLMaskNode: 2,
    GraphCLPerturbEdge: 0,

    # GCC:
    GCCSubgraph: 0,

    # IGSD:
    IGSDPageRankAugment: 0,

    # ours
    AddDumbVariables: 3,
    AddRedundantConstraint: 2,
    ScaleConstraint: 1,
    ScaleObj: 1,
}

__all__ = [
    'GraphCLDropNode',
    'GraphCLMaskNode',
    'GraphCLPerturbEdge',

    'GCCSubgraph',

    'IGSDPageRankAugment',

    'AddRedundantConstraint',
    'AddDumbVariables',
    'ScaleConstraint',
    'ScaleObj',
]