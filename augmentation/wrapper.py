from random import choice, choices
from typing import List, Tuple

from torch_geometric.data import HeteroData


class SingleAugmentWrapper:
    """
    Return 2 views of the graph
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, data: HeteroData) -> HeteroData:
        tf = choice(self.transforms)
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
        t1, t2 = choices(self.transforms, k=2)

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
