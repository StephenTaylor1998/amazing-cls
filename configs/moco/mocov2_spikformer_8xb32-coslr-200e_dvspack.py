_base_ = [
    '../_base_/datasets/dvspack_bs32_mocov2.py',
    '../_base_/schedules/imagenet_sgd_coslr_200e.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='SpikeMoCo',
    queue_len=65536,
    feat_dim=128,
    momentum=0.001,
    backbone=dict(
        type='SpikformerDVS',
        in_channels=2,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        embed_dims=256,
        num_heads=16,
        mlp_ratios=4,
        norm_layer=None,
        depths=2,
        neuron_cfg=dict(
            type='LIFNode',
            v_reset=None,  # Todo: check here {default: v_reset=0.}
            detach_reset=True,  # Todo: check here {default: detach_reset=False}
            surrogate_function=dict(
                type='Sigmoid'
            )
        ),
    ),
    neck=dict(
        type='SpikeMoCoV2Neck',
        in_channels=256,
        hid_channels=512,
        out_channels=128,
        with_avg_pool=False),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.2))

# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
