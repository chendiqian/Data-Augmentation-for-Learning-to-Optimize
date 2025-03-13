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

    # ours, heuristic
    ScaleConstraint: 2,
    ScaleObj: 2,
    ScaleCoordinate: 2,
    AddSubOrthogonalConstraint: 1,
    AddRedundantConstraint: 1,
    DropInactiveConstraint: 1,
    OracleDropInactiveConstraint: 1,
    AddDumbVariables: 3,  # !this must be after scaling!
}
