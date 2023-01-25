from spikingjelly.activation_based.neuron import (
    IFNode, LIFNode, ParametricLIFNode, QIFNode, EIFNode, IzhikevichNode, LIAFNode
)

from .builder import NODES

# IFNode = NODES.register_module(IFNode)
# LIFNode = NODES.register_module(LIFNode)
# ParametricLIFNode = NODES.register_module(ParametricLIFNode)
# QIFNode = NODES.register_module(QIFNode)
# EIFNode = NODES.register_module(EIFNode)
# IzhikevichNode = NODES.register_module(IzhikevichNode)
# LIAFNode = NODES.register_module(LIAFNode)

for node in (IFNode, LIFNode, ParametricLIFNode, QIFNode, EIFNode, IzhikevichNode, LIAFNode):
    NODES.register_module(node)
