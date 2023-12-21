# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, List

import torch
from mmpretrain.models.heads import LinearClsHead
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from torch import nn


@MODELS.register_module()
class SpikeLinearClsHead(LinearClsHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 enable_time_embed=False,
                 time_step=None,
                 init_cfg=None,
                 **kwargs):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)

        super(SpikeLinearClsHead, self).__init__(num_classes, in_channels, init_cfg, **kwargs)
        self.enable_time_embed = enable_time_embed
        if enable_time_embed:
            self.embed = nn.Parameter(torch.ones(time_step))

        self.in_channels = in_channels
        self.num_classes = num_classes
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        if self.enable_time_embed:
            out = self.time_embed(feats[-1])
        else:
            out = feats[-1]
        return out

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats).mean(0)
        cls_score = self.fc(pre_logits)
        return cls_score

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self.fc(self.pre_logits(feats).mean(0))

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def time_embed(self, x):
        assert x.shape[0] == self.embed.shape[0], \
            f'[INFO] [AMZCLS] Check time step. ' \
            f'T1={x.shape[0]} and T2={self.embed.shape[0]}'
        return torch.einsum('i..., i->i...', x, self.embed.sigmoid())


@MODELS.register_module()
class TETLinearClsHead(LinearClsHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 enable_time_embed=False,
                 time_step=None,
                 lamb=1e-3,
                 means=1.0,
                 init_cfg=None,
                 **kwargs):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)

        super(TETLinearClsHead, self).__init__(num_classes, in_channels, init_cfg, **kwargs)
        self.enable_time_embed = enable_time_embed
        if enable_time_embed:
            self.embed = nn.Parameter(torch.ones(time_step))

        self.in_channels = in_channels
        self.num_classes = num_classes
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.lamb = lamb
        self.means = means
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        if self.enable_time_embed:
            out = self.time_embed(feats[-1])
        else:
            out = feats[-1]
        return out

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats).mean(0)
        cls_score = self.fc(pre_logits)
        return cls_score

    def time_embed(self, x):
        assert x.shape[0] == self.embed.shape[0], \
            f'[INFO] [AMZCLS] Check time step. ' \
            f'T1={x.shape[0]} and T2={self.embed.shape[0]}'
        return torch.einsum('i..., i->i...', x, self.embed.sigmoid())

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self.fc(self.pre_logits(feats))
        time, batch, dim = cls_score.shape
        # mse
        y = torch.zeros_like(cls_score).fill_(self.means)
        loss_mse = torch.nn.functional.mse_loss(cls_score, y)

        # cross_entropy
        cls_score = cls_score.view(time * batch, dim)
        data_samples = data_samples * time

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)

        losses['loss'] = losses['loss'] * (1.-self.lamb) + self.lamb * loss_mse
        return losses
