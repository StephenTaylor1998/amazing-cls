import torch
from mmcv.runner import auto_fp16
from spikingjelly.activation_based import layer
from spikingjelly.activation_based.model.spiking_resnet import conv3x3, conv1x1

from .base import BasicBlock, Bottleneck
from ..builder import MODELS
from ..neurons import build_node


@MODELS.register_module()
class PlainDigital(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None, rate=1., neuron_cfg=None, **kwargs):
        super(PlainDigital, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        mid = int(inplanes * rate)
        self.conv1 = conv3x3(inplanes, mid, stride)
        self.bn1 = norm_layer(mid)
        self.sn1 = build_node(neuron_cfg)
        self.conv2 = conv3x3(mid, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = build_node(neuron_cfg)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x',))
    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)
        out = identity + out
        return out


@MODELS.register_module()
class PlainAnalog(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None, rate=1., neuron_cfg=None, **kwargs):
        super(PlainAnalog, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        mid = int(planes * rate)
        self.bn1 = norm_layer(inplanes)
        self.sn1 = build_node(neuron_cfg)
        self.conv1 = conv3x3(inplanes, mid, stride)

        self.bn2 = norm_layer(mid)
        self.sn2 = build_node(neuron_cfg)
        self.conv2 = conv3x3(mid, planes)

        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x',))
    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(self.downsample_sn(x))

        out = self.bn1(x)
        out = self.sn1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.sn2(out)
        out = self.conv2(out)
        out = identity + out
        return out


@MODELS.register_module()
class PlainAnalogV2(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None, rate=1., neuron_cfg=None, **kwargs):
        super(PlainAnalogV2, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        mid = int(planes * rate)

        self.sn1 = build_node(neuron_cfg)
        self.conv1 = conv3x3(inplanes, mid, stride)
        self.bn1 = norm_layer(mid)

        self.sn2 = build_node(neuron_cfg)
        self.conv2 = conv3x3(mid, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x',))
    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(self.downsample_sn(x))

        out = self.sn1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.sn2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = identity + out
        return out


# channel: 64->64->64
@MODELS.register_module()
class BlockA222(PlainDigital):
    def __init__(self, *args, rate=1., **kwargs):
        super(BlockA222, self).__init__(*args, rate=rate, **kwargs)


# channel: 128->32->128
@MODELS.register_module()
class BlockA414(PlainDigital):
    def __init__(self, *args, rate=0.25, **kwargs):
        super(BlockA414, self).__init__(*args, rate=rate, **kwargs)


# channel: 32->128->32
@MODELS.register_module()
class BlockA141(PlainDigital):
    def __init__(self, *args, rate=4., **kwargs):
        super(BlockA141, self).__init__(*args, rate=rate, **kwargs)


# channel: 64->64->64
@MODELS.register_module()
class BlockB222(PlainAnalog):
    def __init__(self, *args, rate=1., **kwargs):
        super(BlockB222, self).__init__(*args, rate=rate, **kwargs)


# channel: 128->32->128
@MODELS.register_module()
class BlockB414(PlainAnalog):
    def __init__(self, *args, rate=0.25, **kwargs):
        super(BlockB414, self).__init__(*args, rate=rate, **kwargs)


# channel: 32->128->32
@MODELS.register_module()
class BlockB141(PlainAnalog):
    def __init__(self, *args, rate=4., **kwargs):
        super(BlockB141, self).__init__(*args, rate=rate, **kwargs)


# channel: 64->64->64
@MODELS.register_module()
class BlockC222(PlainAnalogV2):
    def __init__(self, *args, rate=1., **kwargs):
        super(BlockC222, self).__init__(*args, rate=rate, **kwargs)


# channel: 128->32->128
@MODELS.register_module()
class BlockC414(PlainAnalogV2):
    def __init__(self, *args, rate=0.25, **kwargs):
        super(BlockC414, self).__init__(*args, rate=rate, **kwargs)


# channel: 32->128->32
@MODELS.register_module()
class BlockC141(PlainAnalogV2):
    def __init__(self, *args, rate=4., **kwargs):
        super(BlockC141, self).__init__(*args, rate=rate, **kwargs)


@MODELS.register_module()
class PlainDFBasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None, rate=1., neuron_cfg=None, **kwargs):
        super(PlainDFBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        mid = int(planes * rate)

        self.sn1 = build_node(neuron_cfg)
        self.conv1 = conv3x3(inplanes, mid, stride)
        self.bn1 = norm_layer(mid)

        self.sn2 = build_node(neuron_cfg)
        self.conv2 = conv3x3(mid * 2, planes, groups=2)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = build_node(neuron_cfg)
            self.shortcut_sn = build_node(neuron_cfg)
        self.stride = stride
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x',))
    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(self.downsample_sn(x))

        out = self.sn1(x)
        spike_id = self.shortcut_sn(identity) if self.downsample is not None else out
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.sn2(out)
        # [T, B, C, H, W] or [B, C, H, W]
        out = torch.cat((out, spike_id), dim=-3)
        out = self.conv2(out)
        out = self.bn2(out)

        out = identity + out
        return out


@MODELS.register_module()
class PlainDFBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, neuron_cfg=None):
        super(PlainDFBottleneck, self).__init__()
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
        if downsample is not None:
            self.downsample_sn = build_node(neuron_cfg)
        self.stride = stride
        self.use_res = True
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x',))
    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(self.downsample_sn(x))

        if not self.use_res:
            return identity

        out = self.sn1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.sn2(out)
        out = self.conv2(out)

        out = self.bn2(out)

        out = self.sn3(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        return out
