import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls

from .base import (
    conv1x1, BasicBlock, Bottleneck
)
from ..builder import BACKBONES, MODELS
from ..neurons import build_node, NODES

__all__ = ['DFResNetCifar',

           'df_resnet18c_cifar', 'df_resnet34c_cifar', 'df_resnet50c_cifar',
           'df_resnet101c_cifar', 'df_resnet152c_cifar', 'df_resnext50c_cifar_32x4d',
           'df_resnext101c_cifar_32x8d', 'df_wide_resnet50c_cifar_2', 'df_wide_resnet101c_cifar_2',

           'df_resnet18b_cifar', 'df_resnet34b_cifar', 'df_resnet50b_cifar',
           'df_resnet101b_cifar', 'df_resnet152b_cifar', 'df_resnext50b_cifar_32x4d',
           'df_resnext101b_cifar_32x8d', 'df_wide_resnet50b_cifar_2', 'df_wide_resnet101b_cifar_2',

           'df_resnet18a_cifar', 'df_resnet34a_cifar', 'df_resnet50a_cifar',
           'df_resnet101a_cifar', 'df_resnet152a_cifar', 'df_resnext50a_cifar_32x4d',
           'df_resnext101a_cifar_32x8d', 'df_wide_resnet50a_cifar_2', 'df_wide_resnet101a_cifar_2'
           ]

default_neuron = dict(type='IFNode')
default_width = [64, 128, 256, 512]


@BACKBONES.register_module()
class DFResNetCifar(nn.Module):
    def __init__(self, block_type, layers, width=None, num_classes=10, in_channels=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
                 cnf: str = None, neuron_cfg=None):
        super().__init__()
        block = MODELS.get(block_type)
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if width is None:
            print(f"[INFO] Using default width `{default_width}`.\n"
                  "\tfrom `amzcls.models.backbones.spike_resnet`.")
        if neuron_cfg is None:
            print(f"[INFO] Using default neuron `{default_neuron}`.\n"
                  "\tfrom `amzcls.models.backbones.spike_df_resnet`.")
            neuron_cfg = default_neuron
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.in_width = self.inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # image net: 3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        self.conv1 = layer.Conv2d(in_channels, self.inplanes, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = build_node(neuron_cfg)
        self.layer1 = self._make_layer(
            block, width[0], layers[0], stride=1, cnf=cnf, neuron_cfg=neuron_cfg)
        self.layer2 = self._make_layer(
            block, width[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0], cnf=cnf, neuron_cfg=neuron_cfg)
        self.layer3 = self._make_layer(
            block, width[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1], cnf=cnf, neuron_cfg=neuron_cfg)
        self.layer4 = self._make_layer(
            block, width[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2], cnf=cnf, neuron_cfg=neuron_cfg)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(width[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            print(f"[INFO] zero init residual: `{zero_init_residual}`.\n"
                  "\tfrom `amzcls.models.backbones.spike_df_resnet`.")
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        functional.set_step_mode(self, 'm')
        functional.set_backend(self, backend='cupy', instance=NODES.get(neuron_cfg['type']))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf: str = None, neuron_cfg=None):
        norm_layer = self._norm_layer
        downsample_spike = None
        downsample_poten = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_spike = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

            downsample_poten = nn.Sequential(
                conv1x1(self.in_width, int(planes * (self.base_width / 64.)) * self.groups, stride),
                norm_layer(int(planes * (self.base_width / 64.)) * self.groups),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample_spike, downsample_poten, self.groups,
                            self.base_width, previous_dilation, norm_layer, cnf, neuron_cfg=neuron_cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                dilation=self.dilation, norm_layer=norm_layer, cnf=cnf, neuron_cfg=neuron_cfg))
        self.in_width = int(planes * (self.base_width / 64.))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        functional.reset_net(self)
        x = torch.permute(x, (1, 0, 2, 3, 4))

        potential = self.conv1(x)
        potential = self.bn1(potential)
        spike = self.sn1(potential)

        spike, potential = self.layer1((spike, potential))
        spike, potential = self.layer2((spike, potential))
        spike, potential = self.layer3((spike, potential))
        spike, potential = self.layer4((spike, potential))

        spike = self.avgpool(spike)
        if self.avgpool.step_mode == 's':
            spike = torch.flatten(spike, 1)
        elif self.avgpool.step_mode == 'm':
            spike = torch.flatten(spike, 2)

        spike = self.fc(spike)

        return (spike.mean(0),)

    def forward(self, x):
        return self._forward_impl(x)


def _resnet_cifar(arch, pretrained, progress, **kwargs):
    model = DFResNetCifar(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict.pop('conv1.weight')
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    return model


# Type A
@BACKBONES.register_module()
def df_resnet18a_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet18', pretrained, progress, layers=[2, 2, 2, 2], **kwargs)


@BACKBONES.register_module()
def df_resnet34a_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet34', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def df_resnet50a_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet50', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def df_resnet101a_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet101', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@BACKBONES.register_module()
def df_resnet152a_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet152', pretrained, progress, layers=[3, 8, 36, 3], **kwargs)


@BACKBONES.register_module()
def df_resnext50a_cifar_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet_cifar('resnext50_32x4d', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def df_resnext101a_cifar_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet_cifar('resnext101_32x8d', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@BACKBONES.register_module()
def df_wide_resnet50a_cifar_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet_cifar('wide_resnet50_2', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def df_wide_resnet101a_cifar_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet_cifar('wide_resnet101_2', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


# Type B
@BACKBONES.register_module()
def df_resnet18b_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet18', pretrained, progress, layers=[2, 2, 2, 2], **kwargs)


@BACKBONES.register_module()
def df_resnet34b_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet34', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def df_resnet50b_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet50', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def df_resnet101b_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet101', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@BACKBONES.register_module()
def df_resnet152b_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet152', pretrained, progress, layers=[3, 8, 36, 3], **kwargs)


@BACKBONES.register_module()
def df_resnext50b_cifar_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet_cifar('resnext50_32x4d', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def df_resnext101b_cifar_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet_cifar('resnext101_32x8d', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@BACKBONES.register_module()
def df_wide_resnet50b_cifar_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet_cifar('wide_resnet50_2', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def df_wide_resnet101b_cifar_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet_cifar('wide_resnet101_2', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


# Type C
@BACKBONES.register_module()
def df_resnet18c_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet18', pretrained, progress, layers=[2, 2, 2, 2], **kwargs)


@BACKBONES.register_module()
def df_resnet34c_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet34', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def df_resnet50c_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet50', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def df_resnet101c_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet101', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@BACKBONES.register_module()
def df_resnet152c_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet_cifar('resnet152', pretrained, progress, layers=[3, 8, 36, 3], **kwargs)


@BACKBONES.register_module()
def df_resnext50c_cifar_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet_cifar('resnext50_32x4d', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def df_resnext101c_cifar_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet_cifar('resnext101_32x8d', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)


@BACKBONES.register_module()
def df_wide_resnet50c_cifar_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet_cifar('wide_resnet50_2', pretrained, progress, layers=[3, 4, 6, 3], **kwargs)


@BACKBONES.register_module()
def df_wide_resnet101c_cifar_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet_cifar('wide_resnet101_2', pretrained, progress, layers=[3, 4, 23, 3], **kwargs)
