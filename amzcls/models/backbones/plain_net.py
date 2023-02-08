import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from spikingjelly.activation_based import layer, functional
from spikingjelly.activation_based.model.spiking_resnet import conv3x3

from ..builder import BACKBONES
from ..neurons import build_node, NODES


def spike(inplanes, planes, stride=1, neuron_cfg=None):
    return nn.Sequential(
        conv3x3(inplanes, planes, stride),
        layer.BatchNorm2d(planes),
        build_node(neuron_cfg)
    )


def analog(inplanes, planes, stride=1, neuron_cfg=None):
    return nn.Sequential(
        build_node(neuron_cfg),
        conv3x3(inplanes, planes, stride),
        layer.BatchNorm2d(planes)
    )


def get_by_name(type_name):
    if type_name == 'digital':
        return spike
    elif type_name == 'analog':
        return analog
    else:
        raise NotImplemented


class Block(nn.Module):
    def __init__(self, block_type, inplanes, planes, rate=1., use_res=True, neuron_cfg=None):
        super(Block, self).__init__()
        mid = int(inplanes * rate)
        conv = get_by_name(block_type)
        self.conv1 = conv(inplanes, mid, neuron_cfg=neuron_cfg)
        self.conv2 = conv(mid, planes, neuron_cfg=neuron_cfg)
        self.func = lambda _x, _y: _x + _y if use_res else lambda _x, _y: _x
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x',))
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # out = x + out if use_res else x
        out = self.func(x, out)
        return out


@BACKBONES.register_module()
class PlainNet(nn.Module):
    def __init__(self, in_channel, channels: list, block_in_layers: list, down_samples: list,
                 num_classes, block_type, rate=1., use_res=True, neuron_cfg=None):
        super(PlainNet, self).__init__()
        conv = layer.Conv2d(in_channel, channels[0], (3, 3), padding=1)
        bn = layer.BatchNorm2d(channels[0])
        sn = build_node(neuron_cfg)
        layers = make_layers(channels, block_in_layers, down_samples, block_type, rate, use_res, neuron_cfg)
        if block_type == 'digital':
            self.layers = nn.Sequential(conv, bn, sn, layers)
        elif block_type == 'analog':
            self.layers = nn.Sequential(conv, bn, layers, sn)
        else:
            raise NotImplemented
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(channels[-1], num_classes)
        self.fp16_enabled = False
        functional.set_step_mode(self, 'm')
        functional.set_backend(self, backend='cupy', instance=NODES.get(neuron_cfg['type']))

    @auto_fp16(apply_to=('x',))
    def forward(self, x):
        functional.reset_net(self)
        x = torch.permute(x, (1, 0, 2, 3, 4))

        x = self.layers(x)
        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)

        x = self.fc(x)
        return x.mean(0)


def make_blocks(block_type, inplanes, rate, use_res, neuron_cfg, num_blocks):
    blocks = [Block(block_type, inplanes, inplanes, rate, use_res, neuron_cfg) for _ in range(num_blocks)]
    return nn.Sequential(*blocks)


def make_layers(channels, block_in_layers, down_samples, block_type, rate, use_res, neuron_cfg):
    layers = []
    index, in_channel = 0, 0
    for channel, num_blocks, down_sample in zip(channels, block_in_layers, down_samples):
        if index > 0:
            conv = get_by_name(block_type)
            layers.append(conv(in_channel, channel, down_sample, neuron_cfg))
        layers.append(make_blocks(block_type, channel, rate, use_res, neuron_cfg, num_blocks))
        in_channel = channel
        index += 1
    return nn.Sequential(*layers)
