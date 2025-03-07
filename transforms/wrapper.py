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
        raise NotImplementedError
        for tf_class in self.transforms:
            max_rate = self.max_strength_dict[tf_class]
            tf_class.p = random.random() * max_rate
            data = tf_class(data)

        return data


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


class AnchorAugmentWrapper:
    """
    For now it is only for single PPR transform
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data: HeteroData) -> Tuple[HeteroData, HeteroData]:
        return data, self.transform(data)
