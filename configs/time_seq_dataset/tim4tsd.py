_base_ = [
    '../_base_/datasets/time_seq_dataset.py',
    '../_base_/default_runtime.py'
]

# 88.2500%
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='tim4tsd',
        cnf='add',
        in_channels=1
    ),
    head=dict(
        type='ClsHead',
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-5)
optimizer = dict(type='AdamW', lr=0.01, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='Step',
    step=[64, 128],
    gamma=0.1,
    warmup='linear',
    warmup_iters=500,
)
runner = dict(type='EpochBasedRunner', max_epochs=192)
# fp16 = None

# dataset
dataset_type = 'DVSGesture'
time_step = 16
train_pipeline = [
    # dict(type='RandomCrop', size=32, padding=4),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    # dict(type='TimeSample', keys=['img'], time_step=16, sample_step=12),
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
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=False,
        data_prefix='/hy-tmp/data/dvs-gesture',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=True,
        data_prefix='/hy-tmp/data/dvs-gesture',
        pipeline=test_pipeline,),
    test=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=True,
        data_prefix='/hy-tmp/data/dvs-gesture',
        pipeline=test_pipeline,)
)