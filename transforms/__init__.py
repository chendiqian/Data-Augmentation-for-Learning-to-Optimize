from transforms.trivial import RandomDropNode, RandomMaskNode, RandomDropEdge
from transforms.lp_preserve import (DropInactiveConstraint,
                                    AddRedundantConstraint,
                                    ScaleConstraint, ScaleObj,
                                    AddOrthogonalConstraint)

# priority, the smaller, the higher
TRANSFORM_CODEBOOK = {
    AddRedundantConstraint: 2,
    ScaleConstraint: 1,
    ScaleObj: 1,
}

__all__ = [
    'AddRedundantConstraint',
    'ScaleConstraint',
    'ScaleObj',
]