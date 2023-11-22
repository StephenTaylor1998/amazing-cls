# 2023-01-20
import numpy as np
import torch
from torchvision.transforms import RandomHorizontalFlip, InterpolationMode
from torchvision.transforms import Resize

from amzcls.registry import TRANSFORMS


def to_float_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return torch.FloatTensor(data)
    elif isinstance(data, np.ndarray):
        return torch.FloatTensor(torch.from_numpy(data.copy()))
    elif isinstance(data, int):
        return torch.FloatTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@TRANSFORMS.register_module()
class ToFloatTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_float_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class RandomHorizontalFlipDVS(object):

    def __init__(self, keys, prob=0.5):
        self.keys = keys
        self.flip = RandomHorizontalFlip(p=prob)

    def __call__(self, results):
        for key in self.keys:
            results[key] = self.flip(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class ResizeDVS(object):

    def __init__(self, keys, scale=(48, 48),
                 interpolation=InterpolationMode.BILINEAR):
        self.keys = keys
        self.resize = Resize(scale, interpolation, antialias=True)

    def __call__(self, results):
        for key in self.keys:
            results[key] = self.resize(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class CutoutDVS(object):

    def __init__(self, keys):
        self.keys = keys
        self.min_pool = torch.nn.functional.pool

    def __call__(self, results):
        for key in self.keys:
            results[key] = self.min_pool(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
