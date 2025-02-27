import random
from typing import List, Tuple

from torch_geometric.data import HeteroData


class SingleAugmentWrapper:
    """
    Return 1 views of the graph, perturbation rate can vary
    """

    def __init__(self, transform_class_list: List, rate: float):
        self.transform_class_list = transform_class_list
        self.rate = rate

    def __call__(self, data: HeteroData) -> HeteroData:
        for tf_class in self.transform_class_list:
            tf = tf_class(random.random() * self.rate)
            data = tf(data)
        return data


class DuoAugmentWrapper:
    """
    Return 2 views of the graph
    """

    def __init__(self, transforms: List):
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
        self.transform = transform
        self.negatives = negatives

    def __call__(self, data: HeteroData) -> List[HeteroData]:
        pos = self.transform(data)
        neg = self.transform.neg(data, self.negatives)
        return [data, pos] + neg


class ComboAugmentWrapper:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, data: HeteroData) -> Tuple[HeteroData, HeteroData]:
        d1 = d2 = data
        for tf in self.transforms:
            d1 = tf(d1)
            d2 = tf(d2)
        return d1, d2
