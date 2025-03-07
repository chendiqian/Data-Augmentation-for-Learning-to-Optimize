from transforms.graph_cl import GraphCLDropNode, GraphCLMaskNode, GraphCLPerturbEdge
from transforms.lp_preserve import (DropInactiveConstraint,
                                    OracleDropInactiveConstraint,
                                    AddRedundantConstraint,
                                    ScaleConstraint, ScaleObj, ScaleCoordinate,
                                    AddSubOrthogonalConstraint,
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

    # ours, oracle, for supervised learning
    OracleDropInactiveConstraint: 100,

    # ours, heuristic
    ScaleConstraint: 1,
    ScaleObj: 1,
    ScaleCoordinate: 1,
    AddSubOrthogonalConstraint: 2,
    AddRedundantConstraint: 2,
    DropInactiveConstraint: 2,
    AddDumbVariables: 3,
}
