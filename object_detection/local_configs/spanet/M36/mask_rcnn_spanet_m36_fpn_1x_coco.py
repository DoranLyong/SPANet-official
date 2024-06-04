_base_ = [
    '../../_base_/models/mask_rcnn_spanet_fpn.py',
    '../../_base_/datasets/coco_instance.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_1x.py'
]
# model settings
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='SPANet',
        layers = [6, 6, 18, 6],
        embed_dims = [96, 192, 384, 768],
        init_cfg=dict(type='Pretrained', checkpoint='ckpt/spanet_m36_init.pth.tar')),

    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))

data = dict(samples_per_gpu=4)  # batch_size = num_gpus * samples_per_gpu

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(step=[8, 11])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)
