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
        embed_dims = [64, 128, 320, 512],
        patch_dims = [800//2**2, 800//2**3, 800//2**4, 800//2**5],
        radius=[2**1, 2**1, 2**0, 2**0],   
        init_cfg=dict(type='Pretrained', checkpoint='ckpt/spanet_s36_init.pth.tar')),

    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))

num_gpus=4
samples_per_gpu=2
data = dict(samples_per_gpu=samples_per_gpu)  # batch_size = num_gpus * samples_per_gpu
batch_size=num_gpus*samples_per_gpu
batch_baseline=16

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002*(16/batch_baseline), weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(step=[8, 11])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)
