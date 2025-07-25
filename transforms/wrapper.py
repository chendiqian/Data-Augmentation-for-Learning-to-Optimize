import random
from typing import List, Tuple

import numpy as np
from torch_geometric.data import HeteroData

# from . import TRANSFORM_CODEBOOK


# def sort_transforms_lists(transforms: List):
#     """
#     Sort the transforms list, and merge the ones with the same priority
#     Args:
#         transforms:
#
#     Returns:
#
#     """
#     priors = [TRANSFORM_CODEBOOK[tf.__class__] for tf in transforms]
#     priors = np.unique(priors)
#     prior_reorder_dict = {num: i for i, num in enumerate(priors)}
#
#     prioritized_tf_lists = [[] for _ in range(len(priors))]
#
#     for tf in transforms:
#         prior = TRANSFORM_CODEBOOK[tf.__class__]
#         reordered_prior = prior_reorder_dict[prior]
#         prioritized_tf_lists[reordered_prior].append(tf)
#     return prioritized_tf_lists
#
#
# def sort_transforms(transforms: List):
#     """
#     Sort the transforms list
#     Args:
#         transforms:
#
#     Returns:
#
#     """
#     sorted_tfs = sorted(transforms, key=lambda tf: TRANSFORM_CODEBOOK[tf.__class__])
#     return sorted_tfs


class SingleAugmentWrapper:
    """
    Return 1 views of the graph, perturbation rate can vary
    """

    def __init__(self, transform):
        # self.max_strength_dict = {tf:  for tf in transforms}
        self.max_p = transform.p
        # sorted_transforms_lists = sort_transforms(transforms)
        self.transform = transform

    def __call__(self, data: HeteroData) -> HeteroData:
        self.transform.p = random.random() * self.max_p
        data = self.transform(data)
        return data


class SingleDensityAugmentWrapper:
    """
    Return several views of the graph, for supervised setting. perturbation rate can vary
    """

    def __init__(self, transform, density):
        # self.max_strength_dict = {tf:  for tf in transforms}
        self.max_p = transform.p
        # sorted_transforms_lists = sort_transforms(transforms)
        self.transform = transform
        self.density = density

    def __call__(self, data: HeteroData) -> List[HeteroData]:
        lsts = []
        for _ in range(self.density):
            self.transform.p = random.random() * self.max_p
            lsts.append(self.transform(data))
        return lsts


# class ComboAugmentWrapper:
#     def __init__(self, transforms: List):
#         sorted_transforms_lists = sort_transforms(transforms)
#         self.transforms = sorted_transforms_lists
#
#     def __call__(self, data: HeteroData) -> Tuple[HeteroData, HeteroData]:
#         d1 = d2 = data
#         for tf in self.transforms:
#             # tf1, tf2 = random.choices(tf_list, k=2)
#             d1 = tf(d1)
#             d2 = tf(d2)
#         return d1, d2


class DuoAugmentWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data: HeteroData) -> Tuple[HeteroData, HeteroData]:
        return self.transform(data), self.transform(data)


class AnchorAugmentWrapper:
    """
    For now it is only for single PPR transform
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data: HeteroData) -> Tuple[HeteroData, HeteroData]:
        return data, self.transform(data)
