# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='TS1Net',
        in_channel=3,
        channels=[32, 64, 128, 256],
        block_in_layers=[2, 2, 2, 2],
        down_samples=[2, 2, 2, 2],
        num_classes=10,

        neuron_cfg=dict(
            type='TS1Node',
            # type='IFNode',
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
