import torch
from spikingjelly.activation_based.neuron import (
    LIFNode,
    ParametricLIFNode,
)
from torch import nn

from .module import MS_SPS, MS_Block_Conv
from ..utils.timm import trunc_normal_


class SpikeDrivenTransformer(nn.Module):
    def __init__(
            self,
            img_size_h=128,
            img_size_w=128,
            patch_size=16,
            in_channels=2,
            embed_dims=512,
            num_heads=8,
            mlp_ratios=4,
            depths=2,
            pooling_stat="1111",
            attn_mode="direct_xor",
            spike_mode="lif",
            dvs_mode=False,
    ):
        super().__init__()
        self.depths = depths
        self.dvs = dvs_mode
        patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer_index=j,
                )
                for j in range(depths)
            ]
        )

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.head_lif = ParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, _ = patch_embed(x)
        for blk in block:
            x, _ = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head_lif(x)
        return x,

# if __name__ == '__main__':
#     from spikingjelly.activation_based import functional
#
#     model = SpikeDrivenTransformer(
#         img_size_h=128,
#         img_size_w=128,
#         patch_size=16,
#         in_channels=2,
#         embed_dims=512,
#         num_heads=8,
#         mlp_ratios=4,
#         drop_path_rate=0.0,
#         depths=2,
#         T=4,
#         pooling_stat="1111",
#         attn_mode="direct_xor",
#         spike_mode="lif",
#         dvs_mode=False,
#     ).cuda()
#     functional.set_step_mode(model, 'm')
#     inp = torch.rand((4, 1, 2, 128, 128)).cuda()
#
#     out, = model(inp)
#     print(out.shape)
