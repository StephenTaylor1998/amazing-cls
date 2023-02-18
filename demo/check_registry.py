import torch


from amzcls.models.neurons import builder

# surrogate
cfg = dict(
    type='ATan'
)
surrogate = builder.SURROGATE.build(cfg)
print(surrogate)

# node
cfg = dict(
    type='IFNode',
    surrogate_function=dict(
        type='ATan', alpha=1.0, spiking=True
    )
)
node = builder.build_node(cfg)
print(node)

# backbone
from amzcls.models import build_backbone

cfg = dict(
    type='SEWResNetCifar',
    block_type='SEWBasicBlock',
    layers=[2, 2, 2, 2],
    num_classes=1000,
    in_channels=3,
    zero_init_residual=True,
    groups=1, width_per_group=64,
    replace_stride_with_dilation=None,
    norm_layer=None, cnf='add',
    neuron_cfg=dict(
        type='IFNode',
        surrogate_function=dict(
            type='ATan', alpha=1.0, spiking=True
        )
    ),
)
model = build_backbone(cfg)
inputs = torch.randn((1, 3, 32, 32))
print(model(inputs)[0].shape)

cfg = dict(
        type='spike_preact_resnet18_cifar',
        block_type='SpikePreActBasicBlock',
        num_classes=10,
        in_channels=3,
        zero_init_residual=True,
        groups=1, width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        neuron_cfg=dict(
            type='IFNode',
            surrogate_function=dict(
                type='ATan', alpha=1.0, spiking=True
            )
        ),
    )
model = build_backbone(cfg).cuda()
inputs = torch.randn((1, 1, 3, 32, 32)).cuda()
print(model(inputs)[0].shape)


# dataset
from amzcls.datasets import DATASETS
cfg = dict(
        type='DVSGesture',
        time_step=4,
        data_type='frame',
        split_by='number',
        test_mode=False,
        data_prefix='/hy-tmp/data/dvs-gesture',
        pipeline=[],
    )
dataset = DATASETS.build(cfg)
print(dataset)
data = dataset.load_annotations()
for d in data:
    print(d['img'].shape, d['gt_label'])