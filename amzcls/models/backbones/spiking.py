import torch
import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from spikingjelly.activation_based import layer, functional, surrogate
from spikingjelly.activation_based.neuron import IFNode
from ..builder import BACKBONES


@BACKBONES.register_module()
class SLeNet5(BaseBackbone):
    def __init__(self, num_classes=-1):
        super(SLeNet5, self).__init__()
        self.num_classes = num_classes
        self.sn = IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.features = nn.Sequential(
            layer.Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1)), nn.Tanh(),
            layer.AvgPool2d(kernel_size=2),
            layer.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)), nn.Tanh(),
            layer.AvgPool2d(kernel_size=2),
            layer.Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1)), nn.Tanh())
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                layer.Linear(120, 84),
                nn.Tanh(),
                layer.Linear(84, num_classes),
            )
        functional.set_step_mode(self, 'm')
        functional.set_backend(self, backend='cupy', instance=IFNode)

    def forward(self, x):
        functional.reset_net(self)
        x = torch.permute(x, (1, 0, 2, 3, 4))
        x = self.features(x)
        x = self.sn(x)
        if self.num_classes > 0:
            x = self.classifier(x.squeeze())
        return (x.mean(0),)
