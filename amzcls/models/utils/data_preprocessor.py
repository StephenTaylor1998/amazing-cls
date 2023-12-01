from numbers import Number
from typing import Optional, Sequence

import torch
from mmpretrain.models.utils import ClsDataPreprocessor
from mmpretrain.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class StaticPreprocessor(ClsDataPreprocessor):
    def __init__(self,
                 time_step: int = None,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Number = 0,
                 to_rgb: bool = False,
                 to_onehot: bool = False,
                 num_classes: Optional[int] = None,
                 batch_augments: Optional[dict] = None):
        super(StaticPreprocessor, self).__init__(
            mean=mean, std=std, pad_size_divisor=pad_size_divisor, pad_value=pad_value, to_rgb=to_rgb,
            to_onehot=to_onehot, num_classes=num_classes, batch_augments=batch_augments)
        self.time_step = time_step

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super(StaticPreprocessor, self).forward(data, training)
        data['inputs'] = process_static_data(data['inputs'], self.time_step)
        return data


@MODELS.register_module()
class DVSPreprocessor(ClsDataPreprocessor):
    def __init__(self,
                 time_step: int = None,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Number = 0,
                 to_rgb: bool = False,
                 to_onehot: bool = False,
                 num_classes: Optional[int] = None,
                 batch_augments: Optional[dict] = None):
        super(DVSPreprocessor, self).__init__(
            mean=mean, std=std, pad_size_divisor=pad_size_divisor, pad_value=pad_value, to_rgb=to_rgb,
            to_onehot=to_onehot, num_classes=num_classes, batch_augments=batch_augments)
        self.time_step = time_step

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super(DVSPreprocessor, self).forward(data, training)
        data['inputs'] = process_dvs_data(data['inputs'])
        return data


def process_static_data(data: Tensor, repeats):
    # DATA[B, C, H, W] -> DATA[T, B, C, H, W]
    return torch.repeat_interleave(data.unsqueeze(0), repeats, dim=0)


def process_dvs_data(data: Tensor):
    # DATA[B, T, C, H, W] -> DATA[T, B, C, H, W]
    return torch.transpose(data, 0, 1)
