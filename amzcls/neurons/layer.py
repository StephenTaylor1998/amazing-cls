import math
from functools import partial
from typing import Optional, Union, Callable

import torch
from spikingjelly.activation_based import base, functional
from spikingjelly.activation_based import layer
from torch import Tensor
from torch import adaptive_avg_pool1d
from torch import nn
from torch._C._nn import upsample_linear1d, upsample_nearest1d
from torch.nn import Module, Parameter, init
from torch.nn import functional as F
from torch.nn.common_types import _size_1_t, _size_2_t
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair

from amzcls.neurons import surrogate


class BWConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 surrogate_function: Callable = surrogate.Sigmoid()):
        super(BWConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, device, dtype)
        self.surrogate_function = surrogate_function

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # weight = self.surrogate_function(weight)
        weight = self.surrogate_function(weight) - 0.5
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


class BWLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, surrogate_function: Callable = surrogate.Sigmoid()) -> None:
        super(BWLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.surrogate_function = surrogate_function

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.surrogate_function(self.weight) - 0.5, self.bias)
        # return F.linear(input, self.weight, self.bias)


class LayerNorm(nn.LayerNorm, base.StepModule):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None, step_mode='s'):
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x):
        if self.step_mode == 's':
            return super().forward(x)

        elif self.step_mode == 'm':
            return functional.seq_to_ann_forward(x, super().forward)


class tdBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, channel, *args, **kwargs):
        super(tdBatchNorm2d, self).__init__(channel, *args, **kwargs)
        self.weight.data.mul_(0.5)

    def forward(self, x):
        time, batch, *chw = x.shape
        x = super().forward(x.reshape(time * batch, *chw))
        return x.view(time, batch, *chw).contiguous()


class TimeEfficientBatchNorm2d(nn.Module):
    def __init__(self, out_plane, time_step=1, *args, **kwargs):
        super(TimeEfficientBatchNorm2d, self).__init__()
        self.bn = layer.BatchNorm2d(out_plane, *args, **kwargs)
        self.time_embed = nn.Parameter(torch.ones(time_step))

    def forward(self, x):
        x = self.bn(x) * torch.einsum('i...,i->i...', x, self.time_embed)
        return x


class TimeEfficientLayerNorm(nn.Module):
    def __init__(self, out_plane, time_step=1, *args, **kwargs):
        super(TimeEfficientLayerNorm, self).__init__()
        self.ln = LayerNorm(out_plane, *args, **kwargs)
        self.time_embed = nn.Parameter(torch.ones(time_step))

    def forward(self, x):
        x = self.ln(x) * torch.einsum('i...,i->i...', x, self.time_embed)
        return x


class TemporalSample(nn.Module):
    """
    Down/up samples the input on temoral dimension T.
    support x[T, B, N, ...] or x[B, T, N, ...]
    >>>x = torch.rand(16, 1, 1)  # x[T, B, N, ...]
    >>>torch.nn.functional.interpolate(x.permute(1, 2, 0), size=8, mode='nearest').permute(2, 0, 1)
    >>>TemporalSample(size=8, mode='nearest')(x)
    >>>x = torch.rand(16, 1, 1)  # x[B, T, N, ...]
    >>>torch.nn.functional.interpolate(x.permute(1, 2, 0), size=8, mode='nearest').permute(2, 0, 1)
    >>>TemporalSample(size=8, mode='nearest')(x)
    """

    def __init__(self, size, mode='area', align_corners=False, t_dim=0):
        super(TemporalSample, self).__init__()
        self.size = size
        self.t_dim = t_dim
        if mode == 'area':
            self.pool = adaptive_avg_pool1d
        elif mode == 'linear':
            self.pool = partial(upsample_linear1d, align_corners=align_corners)
            # self.pool = upsample_linear1d
        elif mode == 'nearest':
            self.pool = upsample_nearest1d

    def forward(self, x):
        return self.pool(
            torch.flatten(x, 2).transpose(self.t_dim, 2), self.size,
        ).transpose(self.t_dim, 2).reshape(*x.shape[:self.t_dim], -1, *x.shape[self.t_dim + 1:])


class Conv1dTV(_ConvNd):
    """
        Supported input format X[B, T, C, L]
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_step: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            padding: Union[str, _size_1_t] = 0,
            dilation: _size_1_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None
    ) -> None:
        self.time_step = time_step
        factory_kwargs = {'device': device, 'dtype': dtype}
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels * time_step, out_channels * time_step, kernel_size_, stride_, padding_, dilation_,
            False, _single(0), groups * time_step, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]):
        x = x.flatten(1, 2)
        if self.padding_mode != 'zeros':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = _single(0)
        else:
            padding = self.padding

        out = F.conv1d(x, weight, bias, self.stride, padding, self.dilation, self.groups)
        b, tc, l = out.shape
        return out.reshape(b, self.time_step, tc // self.time_step, l)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)  # [B, T, C, L]


class Conv2dTV(_ConvNd):
    """
        Supported input format X[B, T, C, H, W]
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_step: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None
    ) -> None:
        self.time_step = time_step
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels * time_step, out_channels * time_step, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups * time_step, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]):
        x = x.flatten(1, 2)
        if self.padding_mode != 'zeros':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = _pair(0)
        else:
            padding = self.padding

        out = F.conv2d(x, weight, bias, self.stride, padding, self.dilation, self.groups)
        b, tc, h, w = out.shape
        return out.reshape(b, self.time_step, tc // self.time_step, h, w)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)  # [B, T, C, H, W]


class LinearTV(Module):
    def __init__(self, in_features: int, out_features: int, time_step: int = 1, bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((time_step, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(time_step, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x: Tensor):
        # x[..., T, N_in], weight[T, N_in,  N_out] -> y[..., T, N_out]
        return torch.einsum('...ij,ijk->...ik', x, self.weight) + self.bias

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
