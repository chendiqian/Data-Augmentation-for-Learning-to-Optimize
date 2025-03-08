import random
from typing import List, Tuple

import numpy as np
from torch_geometric.data import HeteroData

from . import TRANSFORM_CODEBOOK


def sort_transforms(transforms: List):
    priors = [TRANSFORM_CODEBOOK[tf.__class__] for tf in transforms]
    priors = np.unique(priors)
    prior_reorder_dict = {num: i for i, num in enumerate(priors)}

    prioritized_tf_lists = [[] for _ in range(len(priors))]

    for tf in transforms:
        prior = TRANSFORM_CODEBOOK[tf.__class__]
        reordered_prior = prior_reorder_dict[prior]
        prioritized_tf_lists[reordered_prior].append(tf)
    return prioritized_tf_lists


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
        sorted_transforms_lists = sort_transforms(transforms)
        self.transforms = sorted_transforms_lists

    def __call__(self, data: HeteroData) -> Tuple[HeteroData, HeteroData]:
        d1 = d2 = data
        for tf_list in self.transforms:
            tf1, tf2 = random.choices(tf_list, k=2)
            d1 = tf1(d1)
            d2 = tf2(d2)
        return d1, d2


class AnchorAugmentWrapper:
    """
    For now it is only for single PPR transform
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data: HeteroData) -> Tuple[HeteroData, HeteroData]:
        return data, self.transform(data)
