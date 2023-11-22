from typing import Optional, Union, Callable

from spikingjelly.activation_based import base, functional
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

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
