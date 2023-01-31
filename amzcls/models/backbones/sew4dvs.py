import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from spikingjelly.activation_based import layer, functional
from spikingjelly.activation_based.neuron import ParametricLIFNode

from ..builder import BACKBONES

SNode = ParametricLIFNode


# todo replace  `nn.Conv2d` with `layer.Conv2d`
def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
        layer.BatchNorm2d(out_channels),
        SNode(detach_reset=True)
    )


def conv1x1(in_channels, out_channels):
    return nn.Sequential(
        layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        layer.BatchNorm2d(out_channels),
        SNode(detach_reset=True)
    )


class SEWBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, connect_f=None):
        super(SEWBlock, self).__init__()
        self.connect_f = connect_f
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            conv3x3(mid_channels, in_channels),
        )
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x',))
    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        if self.connect_f == 'add':
            out = x + out
        elif self.connect_f == 'and':
            out = x * out
        elif self.connect_f == 'iand':
            out = x * (1. - out)
        elif self.connect_f == 'or':
            out = x + out - x * out
        else:
            raise NotImplementedError(self.connect_f)
        # out = x * self.conv(x)
        return out


class PlainBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(PlainBlock, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            conv3x3(mid_channels, in_channels),
        )
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x',))
    def forward(self, x: torch.Tensor):
        return self.conv(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            layer.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            layer.BatchNorm2d(in_channels),
        )
        self.sn = SNode(detach_reset=True)
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x',))
    def forward(self, x: torch.Tensor):
        return self.sn(x + self.conv(x))


@BACKBONES.register_module()
class ResNetN4DVS(nn.Module):
    def __init__(self, layer_list, num_classes, connect_f=None):
        super(ResNetN4DVS, self).__init__()
        self.fp16_enabled = False
        in_channels = 2
        conv = []

        for cfg_dict in layer_list:
            channels = cfg_dict['channels']

            if 'mid_channels' in cfg_dict:
                mid_channels = cfg_dict['mid_channels']
            else:
                mid_channels = channels

            if in_channels != channels:
                if cfg_dict['up_kernel_size'] == 3:
                    conv.append(conv3x3(in_channels, channels))
                elif cfg_dict['up_kernel_size'] == 1:
                    conv.append(conv1x1(in_channels, channels))
                else:
                    raise NotImplementedError

            in_channels = channels

            if 'num_blocks' in cfg_dict:
                num_blocks = cfg_dict['num_blocks']
                if cfg_dict['block_type'] == 'sew':
                    for _ in range(num_blocks):
                        conv.append(SEWBlock(in_channels, mid_channels, connect_f))
                elif cfg_dict['block_type'] == 'plain':
                    for _ in range(num_blocks):
                        conv.append(PlainBlock(in_channels, mid_channels))
                elif cfg_dict['block_type'] == 'basic':
                    for _ in range(num_blocks):
                        conv.append(BasicBlock(in_channels, mid_channels))
                else:
                    raise NotImplementedError

            if 'k_pool' in cfg_dict:
                k_pool = cfg_dict['k_pool']
                conv.append(layer.MaxPool2d(k_pool, k_pool))

        conv.append(layer.Flatten(1))

        self.conv = nn.Sequential(*conv)
        with torch.no_grad():
            x = torch.zeros([1, 2, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, layer.MaxPool2d):
                    x = m(x)
            out_features = x.shape[2:].numel() * in_channels
        self.out = layer.Linear(out_features, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        functional.set_step_mode(self, step_mode='m')
        functional.set_backend(self, backend='cupy', instance=SNode)

    @auto_fp16(apply_to=('x',))
    def forward(self, x: torch.Tensor):
        functional.reset_net(self)
        x = torch.permute(x, (1, 0, 2, 3, 4))

        x = self.conv(x)
        return self.out(x.mean(0))


@BACKBONES.register_module()
def sew4dvs_gesture(cnf: str):
    layer_list = [
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    ]
    num_classes = 11
    return ResNetN4DVS(layer_list, num_classes, cnf)


@BACKBONES.register_module()
def org4dvs_gesture(*args, **kwargs):
    layer_list = [
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
    ]
    num_classes = 11
    return ResNetN4DVS(layer_list, num_classes)


@BACKBONES.register_module()
def spk4dvs_gesture(*args, **kwargs):
    layer_list = [
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
    ]
    num_classes = 11
    return ResNetN4DVS(layer_list, num_classes)


@BACKBONES.register_module()
def sew4dvs_cifar10(cnf):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    ]
    num_classes = 10
    return ResNetN4DVS(layer_list, num_classes, cnf)


@BACKBONES.register_module()
def org4dvs_cifar10(*args, **kwargs):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'plain',
         'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'plain',
         'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'plain',
         'k_pool': 2},
    ]
    num_classes = 10
    return ResNetN4DVS(layer_list, num_classes)


@BACKBONES.register_module()
def spk4dvs_cifar10(*args, **kwargs):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'basic',
         'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'basic',
         'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'basic',
         'k_pool': 2},
    ]
    num_classes = 10
    return ResNetN4DVS(layer_list, num_classes)
