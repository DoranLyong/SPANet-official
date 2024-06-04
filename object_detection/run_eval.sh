config_path=local_configs/fpn_spanet/S24/mask_rcnn_spanet_s24_fpn_1x_coco.py # [mask_rcnn_spanet_s24_fpn_1x_coco.py, retinanet_spanet_s24_fpn_1x_coco.py]
ckpt_path=./work_dirs/mask_rcnn_spanet_s24_fpn_1x_coco/best.pth
num_gpus=1

./tools/dist_test.sh $config_path $ckpt_path $num_gpus --out results.pkl --eval bbox segm

#FORK_LAST3=1 dist_test.sh $config_path $ckpt_path $num_gpus --out results.pkl --eval bbox