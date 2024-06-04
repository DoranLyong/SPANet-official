# Applying SPANet to object detection and instance segmentation



##  Environement Setup

Install [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/tree/v2.28.1).

* Check [mmcv-full version](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip) depending on your pytorch version.

```bash
# Compatible with PyTorch 1.12.1 + CUDA 11.6
pip install timm==0.6.12
pip install -U openmim 
mim install mmcv-full==1.6.0

python setup.py develop
```



## Data preparation

Prepare COCO 2017 according to the [guidelines](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/useful_tools.md#dataset-download) in MMDetection. If you have interest in this dataset, refer to below links:

* [mmdetection docs](https://mmdetection.readthedocs.io/en/stable/1_exist_data_model.html#prepare-datasets)
* [COCO homepage](https://cocodataset.org/#download)



## Evaluation
To evaluate SPANet-S24 + RetinaNet on a single node with 4 GPUs run:

```
#  evaluation by default
FORK_LAST3=1 ./tools/dist_test.sh local_configs/fpn_spanet/S24/retinanet_spanet_s24_fpn_1x_coco.py /path/to/checkpoint_file 4 --out results.pkl --eval bbox

# simple usage 
bash run_eval.sh
```

To evaluate SPANet-S24 + RetinaNet on a single node with 4 GPUs run:

```
#  evaluation by default
./tools/dist_test.sh local_configs/fpn_spanet/S24/mask_rcnn_spanet_s24_fpn_1x_coco.py /path/to/checkpoint_file 4 --out results.pkl --eval bbox segm

# simple usage 
bash run_eval.sh
```



## Training

To train SPANet-S24 + RetinaNet on a single node with 4 GPUs run:

```
#  training by default
FORK_LAST3=1 ./tools/dist_train.sh local_configs/fpn_spanet/S24/retinanet_spanet_s24_fpn_1x_coco.py 4

# simple usage 
bash run_train.sh
```

To train SPANet-S24 + Mask R-CNN on a single node with 4 GPUs run:

```
#  training by default
./tools/dist_train.sh local_configs/fpn_spanet/S24/mask_rcnn_spanet_s24_fpn_1x_coco.py 4

# simple usage 
bash run_train.sh
```



## MACs

To calculate MACs for a model, run:

```
python tools/analysis_tools/get_flops.py /path/to/config

# simple usage
bash get_flops.sh
```



## Bibtex

```
@article{yun2023spanet,
  title={SPANet},
  author={Yun, Guhnoo},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023}
}
```



## Acknowledgment

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[mmdetection](https://github.com/open-mmlab/mmdetection/tree/master), [LITv2](https://github.com/ziplab/LITv2/tree/main/detection), [PoolFormer](https://github.com/sail-sg/poolformer/tree/main/detection).
