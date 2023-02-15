# # optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(policy='step', step=[100, 150])
# runner = dict(type='EpochBasedRunner', max_epochs=200)

# optimizer
optimizer = dict(type='AdamW', lr=0.01, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='Step',
    step=[90, 120],
    gamma=0.1,
    warmup='linear',
    warmup_iters=500,
)
runner = dict(type='EpochBasedRunner', max_epochs=150)
