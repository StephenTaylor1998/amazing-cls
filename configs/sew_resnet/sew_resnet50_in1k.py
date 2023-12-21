_base_ = [
    '../_base_/models/sew_resnet50.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

data_preprocessor = dict(
    type='StaticPreprocessor',
    time_step=4,
)

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
)
