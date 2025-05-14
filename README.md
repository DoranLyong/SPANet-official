# SPANet Official
<p align="left">
<a href="https://arxiv.org/abs/2308.11568" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2308.11568-b31b1b.svg?style=flat" /></a>
<a href="https://openaccess.thecvf.com/content/ICCV2023/html/Yun_SPANet_Frequency-balancing_Token_Mixer_using_Spectral_Pooling_Aggregation_Modulation_ICCV_2023_paper.html" alt="Colab">
    <img src="https://img.shields.io/badge/ICCV_2023-open_access-blue" /></a>
<a href="https://doranlyong.github.io/projects/spanet/"> 
   <img src="https://img.shields.io/badge/project-page-blue"></a>
</p>

### 💬 This repo is the official implementation of:
- ***ICCV2023***: SPANet: Frequency-balancing Token Mixer using Spectral Pooling Aggregation Modulation

### 🗞️ News:
- ***arXiv2025***: [Spectral-Adaptive Modulation Networks for Visual Perception](https://arxiv.org/abs/2503.23947)


### 🤖 It currently includes code and models for the following tasks:
- [x] [Image Classification](./image_classification)
- [x] [Object Detection](object_detection)
- [] [Semantic Segmentation](semantic_segmentation)


## 📖 Introduction
**SPANet** is a new backbone network which can handle the balance problem of high- and low-frequency components for optimal feature representations.


## Main results on ImageNet-1K
Please see [image_classification](image_classification) for more details.

| Model      | Pretrain    | Resolution | Top-1 | #Param. | FLOPs |
| ---------- | ----------- | ---------- | ----- | ------- | ----- |
| SPANet-S   | ImageNet-1K | 224x224    | 83.1  | 28.7M   | 4.6G |
| SPANet-M   | ImageNet-1K | 224x224    | 83.5  | 41.8M   | 6.8G |
| SPANet-MX   | ImageNet-1K | 224x224    | 83.8  | 54.9M   | 9.0G |
| SPANet-B   | ImageNet-1K | 224x224    | 84.0  | 75.9M   | 12.0G |
| SPANet-BX   | ImageNet-1K | 224x224    | 84.4  | 99.8 M   | 15.8G |

## Main results on COCO object detection and instance segmentation 
Please see [object_detection](object_detection) for more details.

### RetinaNet 1x

|         Backbone          | Lr Schd | box mAP | #params |
| :---------------          | :-----  | :-----  |  :----- | 
| SPANet-S                  |   1x    |  43.3   |   38M   | 
| SPANet-M                  |   1x    |  44.0   |   51M   |


### Mask R-CNN 1x

|         Backbone          | Lr Schd | box mAP | mask mAP | #params |
| :---------------          | :-----  | :-----  | :------  | :-----  | 
| SPANet-S                  |   1x    |  44.7   |   40.6   |   48M   | 
| SPANet-M                  |   1x    |  45.2   |   41.0   |  61M    |



## Main results on ADE20K semantice segmentation 
Please see [semantic_segmentation](semantic_segmentation) for more details.

### Semantic FPN

|         Backbone          | Lr Schd | mIoU | #params | FLOPs |
| :------------------- | :----- | :-- | :----- | :--- |
| SPANet-S             |   80K   | 45.4 |   32M | 46G  |
| SPANet-M              |   80K   | 46.2 |   45M | 57G  |



## ⭐ Cite SPANet

If you find this repository useful, please give us stars and use the following BibTeX entry for citation.

```latex
@inproceedings{yun2023spanet,
  title={SPANet: Frequency-balancing Token Mixer using Spectral Pooling Aggregation Modulation},
  author={Yun, Guhnoo and Yoo, Juhan and Kim, Kijung and Lee, Jeongho and Kim, Dong Hwan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6113--6124},
  year={2023}
}
```


## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
