from typing import Callable

import torch
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.neuron import (
    IFNode, LIFNode, ParametricLIFNode, QIFNode, EIFNode, IzhikevichNode, LIAFNode,
    KLIFNode, PSN, MaskedPSN, SlidingPSN, GatedLIFNode
)
from torch import nn

from .builder import NODES

for node in (IFNode, LIFNode, ParametricLIFNode, QIFNode, EIFNode, IzhikevichNode, LIAFNode,
             KLIFNode, PSN, MaskedPSN, SlidingPSN, GatedLIFNode):
    NODES.register_module(module=node)


@NODES.register_module()
class TS1Node(nn.Module):
    def __init__(self, surrogate_function, threshold=1.):
        super(TS1Node, self).__init__()
        self.surrogate_function = surrogate_function
        self.threshold = threshold

    def forward(self, x):
        return self.surrogate_function(x - self.threshold)


@NODES.register_module()
class StateIFNode(IFNode):
    """
    >>>inputs = torch.randn((2, 1, 4))
    >>>node = StateIFNode(inputs.shape)
    >>>out = node(inputs)
    >>>print(node.v)
    tensor([[[ 0.1676, -0.3076, -0.1530, -0.1675]],
            [[-0.0658, -1.4495, -0.3014, -0.2170]]])
    >>>node.reset()
    >>>print(node.v)
    tensor([[[0.4139, 0.1390, 0.8201, 0.3612]],
            [[0.3644, 0.9767, 0.0484, 0.7073]]], grad_fn=<AddBackward0>)
    """

    def __init__(self, init_state_shape=(1, 1, 1, 1), v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
        """
        :param init_state_shape: (B, C, H, W) or (B, N, D)
                        x.shape: (T, B, C, H, W) or (T, B, N, D)
        """
        super(StateIFNode, self).__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.init_state = nn.Parameter(nn.init.uniform_(
            torch.empty(init_state_shape), a=-0.2, b=0.2
        ))
        self.v += self.init_state

    def reset(self):
        super(StateIFNode, self).reset()
        self.v += self.init_state

    def forward(self, *args, **kwargs):
        x = args[0]
        if self.step_mode == 's':
            self.v = torch.broadcast_to(self.v, x.shape).to(x)
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            self.v = torch.broadcast_to(self.v, x[0].shape).to(x)
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)


@NODES.register_module()
class StateLIFNode(LIFNode):
    """
    >>>inputs = torch.randn((2, 1, 4))
    >>>node = StateLIFNode(inputs.shape)
    >>>out = node(inputs)
    >>>print(node.v)
    tensor([[[ 0.1676, -0.3076, -0.1530, -0.1675]],
            [[-0.0658, -1.4495, -0.3014, -0.2170]]])
    >>>node.reset()
    >>>print(node.v)
    tensor([[[0.4139, 0.1390, 0.8201, 0.3612]],
            [[0.3644, 0.9767, 0.0484, 0.7073]]], grad_fn=<AddBackward0>)
    """

    def __init__(self, init_state_shape=(1, 1, 1, 1), tau: float = 2., decay_input: bool = True,
                 v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        :param init_state_shape: (B, C, H, W) or (B, N, D)
                        x.shape: (T, B, C, H, W) or (T, B, N, D)
        """
        super(StateLIFNode, self).__init__(tau, decay_input, v_threshold, v_reset, surrogate_function,
                                           detach_reset, step_mode, backend, store_v_seq)
        self.init_state = nn.Parameter(nn.init.uniform_(
            torch.empty(init_state_shape), a=-0.2, b=0.2
        ))
        self.v += self.init_state

    def reset(self):
        super(StateLIFNode, self).reset()
        self.v += self.init_state

    def forward(self, *args, **kwargs):
        x = args[0]
        if self.step_mode == 's':
            self.v = torch.broadcast_to(self.v, x.shape).to(x)
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            self.v = torch.broadcast_to(self.v, x[0].shape).to(x)
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)
