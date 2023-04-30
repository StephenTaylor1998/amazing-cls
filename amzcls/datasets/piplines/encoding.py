# 2023-01-20
import random

import numpy as np
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class ToTime(object):

    def __init__(self, keys, time_step):
        self.keys = keys
        self.time_step = time_step

    def __call__(self, results):
        for k in self.keys:
            if isinstance(results[k], torch.Tensor):
                results[k] = torch.repeat_interleave(
                    torch.unsqueeze(results[k], 0),
                    repeats=self.time_step, dim=0)
            else:
                results[k] = np.repeat(
                    np.expand_dims(results[k], axis=0),
                    repeats=2, axis=0)

        return results


def to_float_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return torch.FloatTensor(data)
    elif isinstance(data, np.ndarray):
        return torch.FloatTensor(torch.from_numpy(data))
    elif isinstance(data, int):
        return torch.FloatTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@PIPELINES.register_module()
class ToFloatTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_float_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
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
