# dataset settings
dataset_type = 'TinyImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', size=64, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=(64, -1)),
    # dict(type='Resize', size=(64, 64)),
    # dict(type='CenterCrop', crop_size=64),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=512,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_prefix='/hy-tmp/tiny-imagenet-200',
        test_mode=False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/hy-tmp/tiny-imagenet-200',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='/hy-tmp/tiny-imagenet-200',
        test_mode=True,
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='accuracy')
