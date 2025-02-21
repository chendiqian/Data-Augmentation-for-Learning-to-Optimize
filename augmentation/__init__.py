from augmentation.trivial import RandomDropNode, RandomMaskNode, RandomDropEdge
from augmentation.lp_preserve import (DropInactiveConstraint,
                                      AddRedundantConstraint,
                                      ScaleInstance,
                                      AddOrthogonalConstraint)

TRANSFORM_CODEBOOK = {
    '0': RandomDropNode,
    '1': DropInactiveConstraint,
    '2': AddRedundantConstraint,
    '3': ScaleInstance,
    '4': AddOrthogonalConstraint,
    '5': RandomMaskNode,
    '6': RandomDropEdge,
}
