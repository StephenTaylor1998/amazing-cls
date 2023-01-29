from copy import deepcopy

import torch.nn as nn
from spikingjelly.activation_based import layer
from spikingjelly.activation_based.model.spiking_resnet import conv3x3, conv1x1

from ..builder import MODELS
from ..neurons import build_node


class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        pass


class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        pass


def sew_function(cnf: str):
    if cnf == 'add':
        return lambda x, y: x + y
    elif cnf == 'and':
        return lambda x, y: x * y
    elif cnf == 'iand':
        return lambda x, y: x * (1. - y)
    elif cnf == 'or':
        # 1. - ((1. - x) * (1. - y))
        return lambda x, y: x + y - x * y
    else:
        raise NotImplementedError


@MODELS.register_module()
class SEWBasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None,
                 neuron_cfg=None):
        super(SEWBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = build_node(neuron_cfg)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = build_node(neuron_cfg)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.cnf = sew_function(cnf)
        self.use_res = True

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        if not self.use_res:
            return identity

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)
        out = self.cnf(identity, out)
        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


@MODELS.register_module()
class SEWBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None,
                 neuron_cfg=None):
        super(SEWBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = build_node(neuron_cfg)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = build_node(neuron_cfg)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = build_node(neuron_cfg)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.cnf = sew_function(cnf)
        self.use_res = True

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        if not self.use_res:
            return identity

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sn3(out)
        out = self.cnf(identity, out)
        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


@MODELS.register_module()
class SpikePreActBasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, neuron_cfg=None):
        super(SpikePreActBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.sn1 = build_node(neuron_cfg)
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn2 = norm_layer(planes)
        self.sn2 = build_node(neuron_cfg)
        self.conv2 = conv3x3(planes, planes)

        self.downsample = downsample
        self.stride = stride
        self.use_res = True

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.use_res:
            return identity

        out = self.bn1(x)
        out = self.sn1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.sn2(out)
        out = self.conv2(out)
        out += identity
        return out


@MODELS.register_module()
class SpikePreActBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, neuron_cfg=None):
        super(SpikePreActBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.sn1 = build_node(neuron_cfg)
        self.conv1 = conv1x1(inplanes, width)

        self.bn2 = norm_layer(width)
        self.sn2 = build_node(neuron_cfg)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)

        self.bn3 = norm_layer(width)
        self.sn3 = build_node(neuron_cfg)
        self.conv3 = conv1x1(width, planes * self.expansion)

        self.downsample = downsample
        self.stride = stride
        self.use_res = True

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        if not self.use_res:
            return identity

        out = self.bn1(x)
        out = self.sn1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.sn2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.sn3(out)
        out = self.conv3(out)
        out += identity

        return out


@MODELS.register_module()
class DualFlowBasicBlockA(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample_spike=None, downsample_poten=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, neuron_cfg=None):
        super(DualFlowBasicBlockA, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = build_node(neuron_cfg)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = build_node(neuron_cfg)
        self.downsample_spike = deepcopy(downsample_spike)
        self.downsample_poten = deepcopy(downsample_poten)
        if downsample_spike is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.cnf = sew_function(cnf)
        self.use_res = True

    def forward(self, x):
        spike, identity_poten = x
        identity_spike = spike

        if self.downsample_spike is not None:
            identity_spike = self.downsample_sn(self.downsample_spike(identity_spike))
            identity_poten = self.downsample_poten(identity_poten)

        if not self.use_res:
            return identity_spike, identity_poten

        poten = self.conv1(spike)
        poten = self.bn1(poten)
        # identity_poten += poten
        identity_poten = self.cnf(identity_poten, poten)
        spike = self.sn1(identity_poten)

        poten = self.conv2(spike)
        poten = self.bn2(poten)
        spike = self.sn2(poten)

        spike = self.cnf(spike, identity_spike)
        return spike, identity_poten

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


@MODELS.register_module()
class DualFlowBottleneckA(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample_spike=None, downsample_poten=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, neuron_cfg=None):
        super(DualFlowBottleneckA, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = build_node(neuron_cfg)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = build_node(neuron_cfg)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = build_node(neuron_cfg)
        self.downsample_spike = deepcopy(downsample_spike)
        if downsample_spike is not None:
            self.downsample_poten = deepcopy(downsample_poten)
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.cnf = sew_function(cnf)
        self.use_res = True

    def forward(self, x):
        spike, identity_poten = x
        identity_spike = spike

        if self.downsample_spike is not None:
            identity_spike = self.downsample_sn(self.downsample_spike(identity_spike))
            identity_poten = self.downsample_poten(identity_poten)

        if not self.use_res:
            return identity_spike, identity_poten

        poten = self.conv1(spike)
        poten = self.bn1(poten)
        spike = self.sn1(poten)

        poten = self.conv2(spike)
        poten = self.bn2(poten)
        # identity_poten += poten
        identity_poten = self.cnf(identity_poten, poten)
        spike = self.sn2(identity_poten)

        poten = self.conv3(spike)
        poten = self.bn3(poten)
        spike = self.sn3(poten)

        spike = self.cnf(spike, identity_spike)
        return spike, identity_poten

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


@MODELS.register_module()
class DualFlowBasicBlockB(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample_spike=None, downsample_poten=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, neuron_cfg=None):
        super(DualFlowBasicBlockB, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = build_node(neuron_cfg)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = build_node(neuron_cfg)
        self.downsample_spike = deepcopy(downsample_spike)
        self.downsample_poten = deepcopy(downsample_poten)
        if downsample_spike is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.cnf = sew_function(cnf)
        self.use_res = True

    def forward(self, x):
        spike, identity_poten = x
        identity_spike = spike

        if self.downsample_spike is not None:
            identity_spike = self.downsample_sn(self.downsample_spike(identity_spike))
            identity_poten = self.downsample_poten(identity_poten)

        if not self.use_res:
            return identity_spike, identity_poten

        poten = self.conv1(spike)
        # identity_poten = poten = self.bn1(poten + identity_poten)
        identity_poten = self.bn1(self.cnf(identity_poten, poten))
        spike = self.sn1(identity_poten)

        poten = self.conv2(spike)
        poten = self.bn2(poten)
        spike = self.sn2(poten)

        spike = self.cnf(spike, identity_spike)
        return spike, identity_poten

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


@MODELS.register_module()
class DualFlowBottleneckB(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample_spike=None, downsample_poten=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, neuron_cfg=None):
        super(DualFlowBottleneckB, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = build_node(neuron_cfg)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = build_node(neuron_cfg)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = build_node(neuron_cfg)
        self.downsample_spike = deepcopy(downsample_spike)
        self.downsample_poten = deepcopy(downsample_poten)
        if downsample_spike is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.cnf = sew_function(cnf)
        self.use_res = True

    def forward(self, x):
        spike, identity_poten = x
        identity_spike = spike

        if self.downsample_spike is not None:
            identity_spike = self.downsample_sn(self.downsample_spike(identity_spike))
            identity_poten = self.downsample_poten(identity_poten)

        if not self.use_res:
            return identity_spike, identity_poten

        poten = self.conv1(spike)
        poten = self.bn1(poten)
        spike = self.sn1(poten)

        poten = self.conv2(spike)
        identity_poten = self.cnf(identity_poten, poten)
        spike = self.sn2(identity_poten)

        poten = self.conv3(spike)
        poten = self.bn3(poten)
        spike = self.sn3(poten)

        spike = self.cnf(spike, identity_spike)
        return spike, identity_poten

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


@MODELS.register_module()
class DualFlowBasicBlockC(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample_spike=None, downsample_poten=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, neuron_cfg=None):
        super(DualFlowBasicBlockC, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = build_node(neuron_cfg)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = build_node(neuron_cfg)
        self.downsample_spike = deepcopy(downsample_spike)
        self.downsample_poten = deepcopy(downsample_poten)
        if downsample_spike is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.cnf = sew_function(cnf)
        self.use_res = True

    def forward(self, x):
        spike, identity_poten = x
        identity_spike = spike

        if self.downsample_spike is not None:
            identity_spike = self.downsample_sn(self.downsample_spike(identity_spike))
            identity_poten = self.downsample_poten(identity_poten)

        if not self.use_res:
            return identity_spike, identity_poten

        poten = self.conv1(spike)
        poten = self.bn1(poten)
        # poten += identity_poten
        poten = self.cnf(identity_poten, poten)
        spike = self.sn1(poten)

        poten = self.conv2(spike)
        poten = self.bn2(poten)
        spike = self.sn2(poten)

        spike = self.cnf(spike, identity_spike)
        return spike, poten

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


@MODELS.register_module()
class DualFlowBottleneckC(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample_spike=None, downsample_poten=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, neuron_cfg=None):
        super(DualFlowBottleneckC, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = build_node(neuron_cfg)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = build_node(neuron_cfg)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = build_node(neuron_cfg)
        self.downsample_spike = deepcopy(downsample_spike)
        self.downsample_poten = downsample_spike = nn.Sequential(
            conv1x1(inplanes, width),
            norm_layer(width),
        )
        if downsample_spike is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.cnf = sew_function(cnf)
        self.use_res = True

    def forward(self, x):
        spike, identity_poten = x
        identity_spike = spike
        identity_poten = self.downsample_poten(identity_poten)
        if self.downsample_spike is not None:
            identity_spike = self.downsample_sn(self.downsample_spike(identity_spike))

        if not self.use_res:
            return identity_spike, identity_poten

        poten = self.conv1(spike)
        poten = self.bn1(poten)
        # poten += identity_poten
        poten = self.cnf(identity_poten, poten)
        spike = self.sn1(poten)

        poten = self.conv2(spike)
        poten = self.bn2(poten)
        spike = self.sn2(poten)

        poten = self.conv3(spike)
        poten = self.bn3(poten)
        spike = self.sn3(poten)

        spike = self.cnf(spike, identity_spike)
        return spike, poten

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'
