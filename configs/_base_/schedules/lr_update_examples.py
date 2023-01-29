
# lr_config from mmcv.runner.hooks.lr_updater
# for example `lr_config = dict(policy='CosineAnnealing', min_lr_ratio=0.001)`

Fixed = dict(
    policy='Fixed',
    by_epoch=True,
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    warmup_by_epoch=False
)

Step = dict(
    policy='Step',
    step=[30, 60],
    gamma=0.1,
    min_lr=None,
    by_epoch=True,
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

Exp = dict(
    policy='Exp',
    gamma=0.9,
    by_epoch=True,
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

Poly = dict(
    policy='Poly',
    power=1.,
    min_lr=0.,
    by_epoch=True,
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

Inv = dict(
    policy='Inv',
    gamma=0.9,
    power=1.,
    by_epoch=True,
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

CosineAnnealing = dict(
    policy='CosineAnnealing',
    min_lr=None,
    min_lr_ratio=None,
    by_epoch=True,
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

FlatCosineAnnealing = dict(
    policy='FlatCosineAnnealing',
    start_percent=0.75,
    min_lr=None,
    min_lr_ratio=None,
    by_epoch=True,
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

CosineRestart = dict(
    policy='CosineRestart',
    periods=[30, 60],
    restart_weights=[1., 0.5],
    min_lr=None,
    min_lr_ratio=0.001,
    by_epoch=True,
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

Cyclic = dict(
    policy='Cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
    anneal_strategy='cos',
    gamma=1,
    by_epoch=False,
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

OneCycle = dict(
    policy='OneCycle',
    max_lr=0.1,
    total_steps=None,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1e4,
    three_phase=False,
    by_epoch=True,
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

LinearAnnealing = dict(
    policy='LinearAnnealing',
    min_lr=None,
    min_lr_ratio=0.001,
    by_epoch=True,
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    warmup_by_epoch=False
)
