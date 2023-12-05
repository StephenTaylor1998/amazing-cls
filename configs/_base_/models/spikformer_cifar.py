# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SpikformerCifar',
        num_classes=10,
        in_channels=3,
    ),
    head=dict(
        type='SpikeClsHead',
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)

