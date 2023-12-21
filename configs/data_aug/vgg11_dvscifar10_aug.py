_base_ = [
    '../_base_/models/vgg11_dvs.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        neuron_cfg=dict(
            type='LIFNode',
        ),
        in_channels=2,
    ),
    head=dict(
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=10,
            reduction='mean',
            loss_weight=1.0),
        cal_acc=False,
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.5),
        # dict(type='CutMix', alpha=1.0)
    ])
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=0.06,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
    clip_grad=dict(max_norm=1.0),
)
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-8 / 2e-3,
        by_epoch=True,
        end=30,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=30)
]

train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
# train, val, test setting
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)
auto_scale_lr = dict(base_batch_size=16)

# dataset settings

dataset_type = 'DVSCifar10'
time_step = 16
num_bins = 31

augmentation_space = {
    "Identity": ['torch.tensor(0.0)', False],
    "ShearX": ['torch.linspace(-0.3, 0.3, 31)', True],
    "TranslateX": ['torch.linspace(-0.5, 5.0, 31)', True],
    "TranslateY": ['torch.linspace(-0.5, 5.0, 31)', True],
    "Rotate": ['torch.linspace(-30.0, 30.0, 31)', True],
    "Cutout": ['torch.linspace(1.0, 30.0, 31)', True],
}

data_preprocessor = dict(
    type='DVSPreprocessor',
    time_step=time_step,
    num_classes=10,
    to_rgb=False
)

train_pipeline = [
    dict(type='RandomTimeShuffle', keys=['img'], p=0.5),
    dict(type='ToFloatTensor', keys=['img']),
    dict(type='RandomHorizontalFlipDVS', prob=0.5, keys=['img']),
    dict(type='SpikFormerDVS', keys=['img'], augmentation_space=augmentation_space),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='ToFloatTensor', keys=['img']),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=False,
        data_prefix='./data/dvs-cifar10',
        use_ckpt=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=True,
        data_prefix='./data/dvs-cifar10',
        use_ckpt=False,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1,))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
