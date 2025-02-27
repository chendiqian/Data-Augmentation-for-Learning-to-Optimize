import random
from typing import List, Tuple

from torch_geometric.data import HeteroData

from . import TRANSFORM_CODEBOOK


class SingleAugmentWrapper:
    """
    Return 1 views of the graph, perturbation rate can vary
    """

    def __init__(self, transforms: List):
        self.max_strength_dict = {tf: tf.p for tf in transforms}
        sorted_transforms = sorted(transforms, key=lambda tf: TRANSFORM_CODEBOOK[tf.__class__])
        self.transforms = sorted_transforms

    def __call__(self, data: HeteroData) -> HeteroData:
        for tf_class in self.transforms:
            max_rate = self.max_strength_dict[tf_class]
            tf_class.p = random.random() * max_rate
            data = tf_class(data)

        return data


class DuoAugmentWrapper:
    """
    Return 2 views of the graph
    """

    def __init__(self, transforms: List):
        raise DeprecationWarning
        self.transforms = transforms

    def __call__(self, data: HeteroData) -> Tuple[HeteroData, HeteroData]:
        # while True:
        #     # I don't want 2 identity
        #     t1, t2 = choices(self.transforms, k=2)
        #     if not (isinstance(t1, IdentityAugmentation) and isinstance(t2, IdentityAugmentation)):
        #         break
        t1, t2 = random.choices(self.transforms, k=2)

        data1 = t1(data)
        data2 = t2(data)
        return data1, data2


class PosNegAugmentWrapper:
    def __init__(self, transform, negatives):
        raise DeprecationWarning
        self.transform = transform
        self.negatives = negatives

    def __call__(self, data: HeteroData) -> List[HeteroData]:
        pos = self.transform(data)
        neg = self.transform.neg(data, self.negatives)
        return [data, pos] + neg


class ComboAugmentWrapper:
    def __init__(self, transforms: List):
        sorted_transforms = sorted(transforms, key=lambda tf: TRANSFORM_CODEBOOK[tf.__class__])
        self.transforms = sorted_transforms

    def __call__(self, data: HeteroData) -> Tuple[HeteroData, HeteroData]:
        d1 = d2 = data
        for tf in self.transforms:
            d1 = tf(d1)
            d2 = tf(d2)
        return d1, d2
