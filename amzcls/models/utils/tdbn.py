import torch
from torch import nn


class tdBatchNorm(nn.BatchNorm2d):
    def __init__(self, channel):
        super(tdBatchNorm, self).__init__(channel)
        # according to tdBN paper, the initialized weight is changed to alpha*Vth
        self.weight.data.mul_(0.5)

    def forward(self, x):
        B, T, *spatial_dims = x.shape
        out = super().forward(x.reshape(B * T, *spatial_dims))
        BT, *spatial_dims = out.shape
        out = out.view(B, T, *spatial_dims).contiguous()
        return out


# class MinPool2d(nn.Module):
#     def __init__(self, kernel_size=2):
#         super(MinPool2d, self).__init__()
#         self.k = kernel_size
#
#     def forward(self, x):
#         assert x.shape[-1] % self.k == x.shape[-2] % self.k == 0
#         # x[..., H, W] ==> x[..., K, H // k, k, W // k]
#         x = torch.reshape(x, (
#             *x.shape[:-2], x.shape[-2] // self.k, self.k, x.shape[-1] // self.k, self.k
#         ))
#         # x.min(-4).min(-2) ==> x[..., H//k, W//k]
#         x = torch.amin(x, dim=(-1, -3))
#         return x
