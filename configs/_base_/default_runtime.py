# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=50,
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

evaluation = dict(
    interval=1, metric='accuracy',
    metric_options=dict(
        topk=1
    ), save_best='auto')

fp16 = dict(loss_scale='dynamic')
# fp16 = dict(loss_scale=512.)
# fp16 = dict(loss_scale=dict(init_scale=512))
