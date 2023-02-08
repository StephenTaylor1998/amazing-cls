# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PlainNet',
        in_channel=3,
        channels=[32, 32, 32, 32],
        block_in_layers=[2, 2, 2, 2],
        down_samples=[1, 2, 2, 2],
        num_classes=10,
        # block_type='digital',
        block_type='analog',
        rate=1.,
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
    head=dict(
        type='ClsHead',
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
