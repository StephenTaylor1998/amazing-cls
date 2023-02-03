_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/plain_net.py',
    '../_base_/schedules/cifar10_bs128.py'
]

# T1=81.4100%
# T4=85.5600%
model = dict(
    backbone=dict(
        type='PlainNet',
        in_channel=3,
        channels=[64, 64, 64, 64],
        block_in_layers=[2, 2, 2, 2],
        down_samples=[1, 2, 2, 2],
        num_classes=10,
        block_type='analog',
        rate=0.25,
        use_res=True,
        neuron_cfg=dict(
            type='IFNode',
            v_reset=None,
            detach_reset=False,
            surrogate_function=dict(
                type='ATan'
            )
        )
    ),
)

# dataset settings
dataset_type = 'CIFAR10'
time_step = 4
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
        test_mode=True)
)

