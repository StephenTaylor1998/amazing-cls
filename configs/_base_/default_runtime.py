# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]

# custom_hooks=[dict(type='NetResetHook')]

fp16 = dict(loss_scale='dynamic')
# fp16 = dict(loss_scale=512.)
# fp16 = dict(loss_scale=dict(init_scale=512))
