# SPANet for Classification 

<p align="left">
<a href="https://arxiv.org/abs/2308.11568" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2308.11568-b31b1b.svg?style=flat" /></a>
<a href="https://openaccess.thecvf.com/content/ICCV2023/html/Yun_SPANet_Frequency-balancing_Token_Mixer_using_Spectral_Pooling_Aggregation_Modulation_ICCV_2023_paper.html" alt="Colab">
    <img src="https://img.shields.io/badge/ICCV_2023-open_access-blue" /></a>
</p>


### To do:
- [x] Training and validation code 
- [] SPANet checkpoints with demo
- [] Visualization of features out of SPAM



## Requirements

torch>=1.7.0; torchvision>=0.8.0; pyyaml; [timm](https://github.com/rwightman/pytorch-image-models) (`pip install timm==0.6.11`)

Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```


## SPANet 
### Models with the SPAM mixer trained on ImageNet-1K
| Model | Resolution | Params | MACs | Top1 Acc | Download |
| :---     |   :---:    |  :---: |  :---:  |  :---:  |  :---:  |
| SPANet-S | 224 | 29M | 4.6G |  83.1 | |
| SPANet-M | 224 | 42M | 6.8G |  83.5 | |
| SPANet-MX | 224 | 55M | 9.0G |  83.8 | |
| SPANet-B | 224 | 76M | 12.0G |  84.0 |  |
| SPANet-BX | 224 | 100 M | 15.8G | --  |  |


### Validation 
To evaluate our SPANet models, run: 
```bash 
DataPATH=/path/to/imagenet 
MODEL=spanet_medium
ckpt=/path/to/checkpoint 
batch_size=128

python validate.py $DataPATH --model $MODEL -b $batch_size --checkpoint $ckpt
```
You can check an example in [val.sh](./val.sh).

### Train 
We set batch size of 1024 by default and train models with 4 GPUs. For multi-node training, adjust `--grad-accum-steps` depending on your conditions. 

To train (fine-tuning) the models, run:
```bash 
bash ./scripts/spanet/train_spanet_small.sh
```
You can check more details in [scripts](./scripts/spanet).







## Acknowledgement 
Our implementation is mainly based on [metaformer baseline](https://github.com/sail-sg/metaformer). We would like to thank for sharing your nice work!


## Bibtex
```latex
@inproceedings{yun2023spanet,
  title={SPANet: Frequency-balancing Token Mixer using Spectral Pooling Aggregation Modulation},
  author={Yun, Guhnoo and Yoo, Juhan and Kim, Kijung and Lee, Jeongho and Kim, Dong Hwan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6113--6124},
  year={2023}
}
```


