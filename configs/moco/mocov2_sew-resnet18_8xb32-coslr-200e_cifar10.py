_base_ = [
    # '../_base_/datasets/imagenet_bs32_mocov2.py',
    '../_base_/datasets/cifar10_bs32_mocov2.py',
    '../_base_/schedules/imagenet_sgd_coslr_200e.py',
    '../_base_/default_runtime.py',
]

data_preprocessor = dict(
    type='StaticSelfSupDataPreprocessor',
    time_step=4,
)

# model settings
model = dict(
    type='SpikeMoCo',
    queue_len=65536,
    feat_dim=128,
    momentum=0.001,
    backbone=dict(
        type='SEWResNet',
        block_type='SEWBasicBlock',
        layers=[2, 2, 2, 2],
        width=[64, 128, 256, 512],
        stride=[1, 2, 2, 2],
        in_channels=3,
        zero_init_residual=True,
        groups=1, width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        cnf_list=['add'],
        neuron_cfg=dict(
            type='IFNode',
            surrogate_function=dict(
                type='ATan'
            )
        ),
    ),
    neck=dict(
        type='SpikeMoCoV2Neck',
        in_channels=512,
        hid_channels=512,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.2))

# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
