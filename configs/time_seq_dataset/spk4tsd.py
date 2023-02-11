_base_ = [
    '../_base_/datasets/time_seq_dataset.py',
    '../_base_/default_runtime.py'
]

# 88.2500%
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='spk4tsd',
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

