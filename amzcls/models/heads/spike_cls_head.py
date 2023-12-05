# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from mmpretrain.models.heads import ClsHead
from mmpretrain.registry import MODELS
from torch import nn


@MODELS.register_module()
class SpikeClsHead(ClsHead):

    def __init__(self,
                 trainable=False,
                 time_step=None,
                 loss=None,
                 topk: Union[int, Tuple[int]] = (1,),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        if loss is None:
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        super(SpikeClsHead, self).__init__(loss, topk, cal_acc, init_cfg)
        self.trainable = trainable
        if trainable:
            self.time_embed = nn.Parameter(torch.ones(time_step))

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        if self.trainable:
            assert feats[-1].shape[0] == self.time_embed.shape[0], \
                f'[INFO] [AMZCLS] Check time step. ' \
                f'T1{feats[-1].shape[0]} and T2{self.time_embed.shape[0]}'
            out = torch.einsum('i..., i->...', feats[-1], self.time_embed.sigmoid())
            # out = torch.einsum('i..., i->...', feats[-1], self.time_embed)
        else:
            out = feats[-1].mean(0)
        # print(self.time_embed[:, 0, 0])  # todo delete this
        return out

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats)
        return pre_logits


@MODELS.register_module()
class LogClsHead(ClsHead):

    def __init__(self,
                 time_step: int,
                 trainable=False,
                 loss=None,
                 topk: Union[int, Tuple[int]] = (1,),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        if loss is None:
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        super(LogClsHead, self).__init__(loss, topk, cal_acc, init_cfg)
        self.time_embed = nn.Parameter(logarithmic(time_step), requires_grad=trainable)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        assert feats[-1].shape[0] == self.time_embed.shape[0], \
            f'[INFO] [AMZCLS] Check time step. ' \
            f'T1{feats[-1].shape[0]} and T2{self.time_embed.shape[0]}'
        out = torch.einsum('i..., i->...', feats[-1], self.time_embed)
        return out

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats)
        return pre_logits


@MODELS.register_module()
class MovingAverageHead(ClsHead):

    def __init__(self,
                 time_step: int,
                 trainable=False,
                 beta=0.3,
                 loss=None,
                 topk: Union[int, Tuple[int]] = (1,),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        if loss is None:
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        super(MovingAverageHead, self).__init__(loss, topk, cal_acc, init_cfg)
        self.time_embed = nn.Parameter(maea(time_step, beta), requires_grad=trainable)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        assert feats[-1].shape[0] == self.time_embed.shape[0], \
            f'[INFO] [AMZCLS] Check time step. ' \
            f'T1{feats[-1].shape[0]} and T2{self.time_embed.shape[0]}'
        out = torch.einsum('i..., i->...', feats[-1], self.time_embed)
        return out

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats)
        return pre_logits


@MODELS.register_module()
class FixClsHead(ClsHead):

    def __init__(self,
                 time_embed: list,
                 trainable=False,
                 loss=None,
                 topk: Union[int, Tuple[int]] = (1,),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        if loss is None:
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        super(FixClsHead, self).__init__(loss, topk, cal_acc, init_cfg)
        self.time_embed = nn.Parameter(torch.tensor(time_embed), requires_grad=trainable)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        assert feats[-1].shape[0] == self.time_embed.shape[0], \
            f'[INFO] [AMZCLS] Check time step. ' \
            f'T1{feats[-1].shape[0]} and T2{self.time_embed.shape[0]}'
        return torch.einsum('i..., i->...', feats[-1], self.time_embed)

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats)
        return pre_logits


def check_time_step(time_embed, length):
    # todo: impl VIT like interpolate.
    if length == time_embed.shape[0]:
        return time_embed
    else:
        return torch.nn.functional.interpolate(time_embed, size=length)


def logarithmic_fixed(length, end=16.):
    linespace = torch.linspace(start=1., end=end, steps=length)
    out = torch.log(linespace)
    return out / out.sum()


def logarithmic(length):
    linespace = torch.range(start=1, end=length, step=1)
    out = torch.log(linespace)
    return out / out.sum()


def maea(length, beta=0.3):
    """
    Moving Average Expansion Approximate
    approximate: [b*(1-b)^(length-1-i) for i in range(length)]
    """
    exp1 = torch.range(start=length - 1, end=0, step=-1)
    exp2 = torch.ones(length)
    return torch.pow(beta, exp2) * torch.pow((1 - beta), exp1)


def mae(length, beta=0.3):
    """
    Moving Average Expansion
    length=2 ->                                (1-b), b
    length=3 ->                      (1-b)^2, b(1-b), b
    length=4 ->            (1-b)^3, b(1-b)^2, b(1-b), b
    """
    exp1 = torch.range(start=length - 1, end=0, step=-1)
    exp2 = torch.concat([torch.zeros(1), torch.ones(length - 1)])
    return torch.pow(beta, exp2) * torch.pow((1 - beta), exp1)
