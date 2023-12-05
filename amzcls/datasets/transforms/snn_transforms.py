# 2023-01-20
import random

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
class TimeSample(object):

    def __init__(self, keys, time_step: int, sample_step: int, use_rand=True):
        self.keys = keys
        self.time_step = time_step
        self.sample_step = sample_step
        self.use_rand = use_rand

    def __call__(self, results):
        sample_step = random.randint(self.sample_step, self.time_step) if self.use_rand else self.sample_step
        indices = random.sample(
            range(self.time_step), sample_step
        )
        indices.sort()
        for k in self.keys:
            results[k] = results[k][indices]

            if self.use_rand:
                zero = np.zeros((self.time_step - sample_step, *results[k].shape[1:]), dtype=results[k].dtype)
                results[k] = np.concatenate((results[k], zero), axis=0)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class RandomTimeShuffleLegacy(object):
    # TODO: will be deprecated.
    def __init__(self, keys, time_step: int, p=0.5):
        self.keys = keys
        self.time_step = time_step
        self.p = p

    def __call__(self, results):
        if np.random.rand(1) > self.p:
            return results

        indices = np.arange(self.time_step)
        np.random.shuffle(indices)
        for k in self.keys:
            results[k] = results[k][indices]

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class RandomTimeShuffle(object):

    def __init__(self, keys, p=0.5):
        self.keys = keys
        self.p = p

    def __call__(self, results):
        if np.random.rand(1) > self.p:
            return results

        for k in self.keys:
            time_step = results[k].shape[0]
            indices = np.arange(time_step)
            np.random.shuffle(indices)
            results[k] = results[k][indices]

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class RandomSliding(object):
    """
        {x: [1, 2, 3, 4, 5], sliding: +2} -> [3, 4, 5, 0, 0]
        {x: [1, 2, 3, 4, 5], sliding: -2} -> [0, 0, 1, 2, 3]
    """

    def __init__(self, keys, max_sliding: int, p=0.5):
        self.keys = keys
        self.max_sliding = max_sliding
        self.p = p

    def __call__(self, results):
        if np.random.rand(1) > self.p:
            return results

        for k in self.keys:
            sliding = np.random.randint(-self.max_sliding, self.max_sliding + 1)
            front_part = results[k][:sliding]
            rear_part = results[k][sliding:]
            if sliding == 0:
                return results
            elif sliding > 0:
                results[k] = np.concatenate((rear_part, np.zeros_like(front_part)), axis=0)
            else:
                results[k] = np.concatenate((np.zeros_like(rear_part), front_part), axis=0)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class RandomCircularSliding(object):
    """
        {x: [1, 2, 3, 4, 5], sliding: +2} -> [3, 4, 5, 1, 2]
        {x: [1, 2, 3, 4, 5], sliding: -2} -> [4, 5, 1, 2, 3]
    """

    def __init__(self, keys, max_sliding: int, p=0.5):
        self.keys = keys
        self.max_sliding = max_sliding
        self.p = p

    def __call__(self, results):
        if np.random.rand(1) > self.p:
            return results

        for k in self.keys:
            sliding = np.random.randint(
                -self.max_sliding, self.max_sliding + 1)
            results[k] = np.concatenate(
                (results[k][sliding:], results[k][:sliding]), axis=0
            )

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
