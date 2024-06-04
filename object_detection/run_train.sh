config_path=local_configs/spanet/S24/mask_rcnn_spanet_s24_fpn_1x_coco_r1100.py   # [mask_rcnn_spanet_s24_fpn_1x_coco.py, retinanet_spanet_s24_fpn_1x_coco.py]
num_gpus=4

./tools/dist_train.sh $config_path $num_gpus  
#FORK_LAST3=1 ./tools/dist_train.sh $config_path $num_gpus

