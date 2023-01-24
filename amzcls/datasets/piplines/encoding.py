# 2023-01-20
import torch
import numpy as np

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
