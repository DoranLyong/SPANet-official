_base_ = [
    '../../_base_/models/retinanet_spanet_fpn.py',
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_1x.py'
]
# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='SPANet',
        layers = [4, 4, 12, 4],
        embed_dims = [64, 128, 320, 512],
        init_cfg=dict(type='Pretrained', checkpoint='ckpt/spanet_s24_init.pth.tar')),

    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))

data = dict(samples_per_gpu=4)  # batch_size = num_gpus * samples_per_gpu

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(step=[8, 11])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)