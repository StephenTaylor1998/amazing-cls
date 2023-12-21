_base_ = [
    '../_base_/models/spikformer_cifar.py',
    '../_base_/datasets/cifar10_bs256_rsb_a12.py',
    '../_base_/schedules/imagenet_bs2048_rsb.py',
    '../_base_/default_runtime.py'
]

data_preprocessor = dict(
    type='StaticPreprocessor',
    time_step=4,
)

# model settings
model = dict(
    backbone=dict(
        type='SpikformerCifar',
        in_channels=3,
    ),
    head=dict(
        # loss=dict(use_sigmoid=True),
        num_classes=10,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=10,
            reduction='mean',
            loss_weight=1.0),
        cal_acc=False,
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.1),
        dict(type='CutMix', alpha=1.0)
    ]))

# dataset settings
train_dataloader = dict(sampler=dict(type='RepeatAugSampler', shuffle=True))

# schedule settings
optim_wrapper = dict(
    optimizer=dict(lr=0.005),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=300)
]

train_cfg = dict(by_epoch=True, max_epochs=300)
