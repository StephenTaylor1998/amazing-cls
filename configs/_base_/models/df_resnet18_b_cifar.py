# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DFResNetCifar',
        block_type='DualFlowBasicBlockB',
        layers=[2, 2, 2, 2],
        width=[64, 128, 256, 512],
        num_classes=10,
        in_channels=3,
        zero_init_residual=True,
        groups=1, width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        cnf='add',
        neuron_cfg=dict(
            type='IFNode',
            v_reset=None,
            detach_reset=False,
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
