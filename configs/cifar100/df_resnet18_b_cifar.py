_base_ = [
    '../_base_/models/df_resnet18_b_cifar.py', '../_base_/datasets/cifar100_bs512.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        num_classes=100,
    )
)
