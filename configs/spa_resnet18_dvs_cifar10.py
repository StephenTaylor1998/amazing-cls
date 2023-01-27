_base_ = [
    './_base_/models/spa_resnet18_cifar.py', './_base_/datasets/dvs_cifar10_t4.py',
    './_base_/schedules/cifar10_bs128.py', './_base_/default_runtime.py'
]

# 88.2500%
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SpikePreActResNetCifar',
        block_type='SpikePreActBasicBlock',
        layers=[2, 2, 2, 2],
        width=[64, 128, 256, 512],
        num_classes=11,
        in_channels=2,
        zero_init_residual=True,
        groups=1, width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        neuron_cfg=dict(
            type='IFNode',
            surrogate_function=dict(
                type='ATan'
            )
        ),
    ),
    head=dict(
        type='ClsHead',
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)


