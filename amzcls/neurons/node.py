# from mmcv.runner import auto_fp16
from spikingjelly.activation_based.neuron import (
    IFNode, LIFNode, ParametricLIFNode, QIFNode, EIFNode, IzhikevichNode, LIAFNode, KLIFNode
)
from torch import nn

from .builder import NODES

for node in (IFNode, LIFNode, ParametricLIFNode, QIFNode, EIFNode, IzhikevichNode, LIAFNode, KLIFNode):
    NODES.register_module(module=node)


@NODES.register_module()
class TS1Node(nn.Module):
    def __init__(self, surrogate_function, threshold=1.):
        super(TS1Node, self).__init__()
        self.surrogate_function = surrogate_function
        self.threshold = threshold
        self.fp16_enabled = False

    # @auto_fp16(apply_to=('x',))
    def forward(self, x):
        # return torch.gt(x, self.threshold).to(x)
        return self.surrogate_function(x - self.threshold)
