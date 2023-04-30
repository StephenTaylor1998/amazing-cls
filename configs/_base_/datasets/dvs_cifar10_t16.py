# dataset settings
dataset_type = 'DVSCifar10'
time_step = 16

train_pipeline = [
    # dict(type='RandomCrop', size=32, padding=4),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='TimeSample', keys=['img'], time_step=16, sample_step=12),
    dict(type='ToFloatTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='SNNAugment', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='ToFloatTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=False,
        data_prefix='/hy-tmp/data/dvs-cifar10',
        use_ckpt=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=True,
        data_prefix='/hy-tmp/data/dvs-cifar10',
        use_ckpt=True,
        pipeline=test_pipeline, ),
    test=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=True,
        data_prefix='/hy-tmp/data/dvs-cifar10',
        use_ckpt=True,
        pipeline=test_pipeline, )
)
