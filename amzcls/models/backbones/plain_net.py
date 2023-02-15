from mmcv.runner import auto_fp16
from spikingjelly.activation_based import layer
from spikingjelly.activation_based.model.spiking_resnet import conv3x3

from .base import BasicBlock
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
        mid = int(inplanes * rate)
        self.bn1 = norm_layer(inplanes)
        self.sn1 = build_node(neuron_cfg)
        self.conv1 = conv3x3(inplanes, mid, stride)

        self.bn2 = norm_layer(mid)
        self.sn2 = build_node(neuron_cfg)
        self.conv2 = conv3x3(mid, planes)

        self.downsample = downsample
        self.stride = stride
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x',))
    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.bn1(x)
        out = self.sn1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.sn2(out)
        out = self.conv2(out)
        out = identity + out
        return out


# channel: 64->64->64
@MODELS.register_module()
class PlainDigitalA(PlainDigital):
    def __init__(self, *args, rate=1., **kwargs):
        super(PlainDigitalA, self).__init__(*args, rate=rate, **kwargs)


# channel: 128->32->128
@MODELS.register_module()
class PlainDigitalB(PlainDigital):
    def __init__(self, *args, rate=0.25, **kwargs):
        super(PlainDigitalB, self).__init__(*args, rate=rate, **kwargs)


# channel: 32->128->32
@MODELS.register_module()
class PlainDigitalC(PlainDigital):
    def __init__(self, *args, rate=4., **kwargs):
        super(PlainDigitalC, self).__init__(*args, rate=rate, **kwargs)


# channel: 64->64->64
@MODELS.register_module()
class PlainAnalogA(PlainAnalog):
    def __init__(self, *args, rate=1., **kwargs):
        super(PlainAnalogA, self).__init__(*args, rate=rate, **kwargs)


# channel: 128->32->128
@MODELS.register_module()
class PlainAnalogB(PlainAnalog):
    def __init__(self, *args, rate=0.25, **kwargs):
        super(PlainAnalogB, self).__init__(*args, rate=rate, **kwargs)


# channel: 32->128->32
@MODELS.register_module()
class PlainAnalogC(PlainAnalog):
    def __init__(self, *args, rate=4., **kwargs):
        super(PlainAnalogC, self).__init__(*args, rate=rate, **kwargs)
