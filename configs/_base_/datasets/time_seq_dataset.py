# dataset settings
dataset_type = 'TimeSeqDataset'
time = 16
sample_time = 12

train_pipeline = [
    dict(type='ToFloatTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
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
        time=time,
        sample_time=sample_time,
        h=time,
        w=time,
        test_rate=0.5,
        test_mode=False,
        data_prefix='None',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        time=time,
        sample_time=sample_time,
        h=time,
        w=time,
        test_rate=0.5,
        test_mode=True,
        data_prefix='None',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        time=time,
        sample_time=sample_time,
        h=time,
        w=time,
        test_rate=0.5,
        test_mode=True,
        data_prefix='None',
        pipeline=test_pipeline)
)
