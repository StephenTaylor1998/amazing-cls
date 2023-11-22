# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Spikformer',
        num_classes=10,
        in_channels=2,
    ),
    head=dict(
        type='ClsHead',
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)

