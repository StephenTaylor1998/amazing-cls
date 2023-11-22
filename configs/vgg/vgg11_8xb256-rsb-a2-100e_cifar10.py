_base_ = [
    '../_base_/models/vgg11_dvs.py',
    '../_base_/datasets/cifar10_bs256_rsb_a12.py',
    '../_base_/schedules/imagenet_bs2048_rsb.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    head=dict(
        loss=dict(use_sigmoid=True),
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
    optimizer=dict(lr=0.008),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))
