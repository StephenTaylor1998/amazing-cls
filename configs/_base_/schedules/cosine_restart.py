# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineRestart', periods=[30, 60, 200],
    restart_weights=[1., 0.9, 0.6],
    min_lr=None,
    min_lr_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=200)

# custom_hooks=[dict(type='NetResetHook')]
