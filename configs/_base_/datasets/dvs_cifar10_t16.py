import torch

# dataset settings
dataset_type = 'DVSCifar10'
time_step = 16

num_bins = 31
augmentation_space = {
    # op_name: (magnitudes, signed)
    "Identity": (torch.tensor(0.0), False),
    "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
    "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
    "TranslateX": (torch.linspace(0.0, 5.0, num_bins), True),
    "TranslateY": (torch.linspace(0.0, 5.0, num_bins), True),
    "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
    "Cutout": (torch.linspace(0.0, 30.0, num_bins), True),
}
# augmentation_space = {
#     # op_name: (magnitudes, signed)
#     "Identity": (torch.tensor(0.0), False),
#     "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
#     "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
#     "TranslateX": (torch.linspace(0.0, 20.0, num_bins), True),
#     "TranslateY": (torch.linspace(0.0, 20.0, num_bins), True),
#     "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
#     "Cutout": (torch.linspace(0.0, 30.0, num_bins), True),
#     "Brightness": (torch.linspace(0.0, 0.3, num_bins), True),
#     "Color": (torch.linspace(0.0, 0.3, num_bins), True),
#     "Contrast": (torch.linspace(0.0, 0.3, num_bins), True),
#     "Sharpness": (torch.linspace(0.0, 0.3, num_bins), True),
# }
train_pipeline = [
    dict(type='TimeSample', keys=['img'], time_step=16, sample_step=12, use_rand=False),
    dict(type='ToFloatTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='SNNAugment', keys=['img'], augmentation_space=augmentation_space),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='ToFloatTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=False,
        data_prefix='/hy-tmp/data/dvs-cifar10',
        use_ckpt=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=True,
        data_prefix='/hy-tmp/data/dvs-cifar10',
        use_ckpt=True,
        pipeline=test_pipeline, ),
    test=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=True,
        data_prefix='/hy-tmp/data/dvs-cifar10',
        use_ckpt=True,
        pipeline=test_pipeline, )
)
