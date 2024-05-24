_base_ = [
    '../_base_/models/sew_resnet50.py',
    '../_base_/datasets/imagenet_bs64.py',
    # '../_base_/datasets/imagenet_bs128_rsb_a3.py',
    '../_base_/default_runtime.py'
]

data_preprocessor = dict(
    type='StaticPreprocessor',
    time_step=4,
)

model = dict(
    backbone=dict(neuron_cfg=dict(type='IFNode', v_reset=None)),
    head=dict(num_classes=1000, out_time_step=[3, ])
)

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='Lamb', lr=0.005, weight_decay=0.02))

# learning policy
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
        T_max=95,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=100)
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
# train, val, test setting
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
auto_scale_lr = dict(base_batch_size=48)

train_dataloader = dict(
    batch_size=48,
    num_workers=16,
    pin_memory=True,
    pin_memory_device="cuda"
)

val_dataloader = dict(
    batch_size=48,
    num_workers=16,
    pin_memory=True,
    pin_memory_device="cuda"
)
