_base_ = [
    '../../_base_/schedules/cifar10_bs128.py',
    '../../_base_/default_runtime.py'
]

# Channel[64, 128, 256, 512] Block[2, 2, 2, 2] BottleRate=1.00 T=1
time_step = 1
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SpikePreActResNetCifar',
        block_type='BlockB222',
        layers=[6, 6, 6, 6],
        width=[64, 128, 256, 512],
        stride=[1, 2, 2, 2],
        num_classes=10,
        in_channels=3,
        zero_init_residual=True,
        groups=1, width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        neuron_cfg=dict(
            type='IFNode',
            detach_reset=True,
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

# dataset settings
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='ToTime', keys=['img'], time_step=time_step),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTime', keys=['img'], time_step=time_step),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=512,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True))
