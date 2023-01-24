# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SLeNet5',
        num_classes=10,
    ),
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ClsHead',
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
