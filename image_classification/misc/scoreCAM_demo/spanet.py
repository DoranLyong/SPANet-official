# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SPANet including small, meidum, and base models.
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models).
"""
import copy

from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers.helpers import to_2tuple


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'spanet_s': _cfg(crop_pct=0.9),
    'spanet_m': _cfg(crop_pct=0.9),
    'spanet_b': _cfg(crop_pct=0.95),
}


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class SPF(nn.Module):
    """ Spectral Pooling Filter 
    """
    def __init__(self, H, W, r, lamb): 
        super().__init__() 
        self.filter = nn.Parameter(self._CircleFilter(H, W, r, lamb).unsqueeze(0), requires_grad=False) # (1,H,W) with no_grad

    def _CircleFilter(self, H, W, r, lamb): 
        """ 
            --image size (H,W)
            --r : radius 
        """
        x_center = int(W//2)
        y_center = int(H//2)
        X, Y = torch.meshgrid(torch.arange(0, H, 1), torch.arange(0,W,1)) 
        circle = torch.sqrt((X-x_center)**2 + (Y-y_center)**2) 

        lp_F = (circle < r).clone().to(torch.float32)
        hp_F = (circle > r).clone().to(torch.float32)

        combined_Filter = lp_F*lamb + hp_F*(1-lamb)  # (H, W)
        combined_Filter[ ~(circle < r) & ~(circle > r)] = 1/3 # cutoff 

        return combined_Filter 

    def _shift(self, x): 
        """ shift Fourier transformed feature map
            then, low_frequency components are at the center
            --x : (B,C,H,W)
        """
        return torch.fft.fftshift(x)

    def _ishift(self, x): 
        """ inverted shift Fourier transformed feature map
            then, low_frequency components are at the corner 
            --x : (B,C,H,W)
        """
        return torch.fft.ifftshift(x)

    def forward(self, x):
        _,_,in_H,in_W = x.shape
        dtype = x.dtype 

        # -- FFT with shift -- 
        x = self._shift(torch.fft.fft2(x, dim=(2, 3), norm='ortho')) # (B, C, in_H, in_W)

        # -- filtering -- 
        _, f_H, f_W = self.filter.shape
        if (in_H, in_W) == (f_H, f_W):
            # if input size is the same as the size of the filter,
            x = self.filter * x 
        else:
            pad_h = in_H - f_H
            pad_w = in_W  - f_W
            padding = (pad_w // 2 + pad_w%2, pad_w // 2, pad_h // 2 + pad_h%2, pad_h // 2)  # (pad_left, pad_right, pad_top, pad_bottom)
            self.filter = F.pad(self.filter, padding, mode='constant', value=self.filter[0, 0 ,0])
            x = self.filter * x 
        
        # --- iFFT with inverse shift --- 
        x = torch.fft.ifft2(self._ishift(x), s=(in_H,in_W), dim=(2,3), norm='ortho').real.to(dtype)
        return x


class SPAM(nn.Module):
    """
    Implementation of Spectral Pooling Aggregation Module for SPANet
    --dim: embeding dim size 
    --k_size: kernel size 
    --r : radius 
    """
    def __init__(self, dim= 64, k_size=7, H=224, W=224, r=2**5):
        super().__init__()
        self.lambs = [lamb for lamb in torch.arange(0.7, 0.9, 0.1)] # (start, end, step)
        print(self.lambs)

        self.n_chunk = len(self.lambs)
        dims = [size.shape[-1] for size in torch.chunk(torch.ones(dim), len(self.lambs), dim=-1)]
        

        self.proj_in = nn.Conv2d(dim, dim, 1) # pw-conv for channel expansion
        self.conv =  nn.Sequential(            
                nn.Conv2d(dim, dim, (1,k_size), padding=(0, k_size//2), groups=dim),
                nn.Conv2d(dim, dim, (k_size,1), padding=(k_size//2, 0), groups=dim),
            )
        self.proj_out = nn.Conv2d(dim, dim, 1) 

        self.sps = nn.ModuleList(
            [SPF(H, W, r, lamb) for lamb in self.lambs]
        )
        self.pws = nn.ModuleList(
            [nn.Conv2d(dims[i], dim, 1) # 1x1 pw-conv
            for i in range(len(self.lambs))]
        )
    
    def forward(self, x):
        # == Token interaction == # 
        x = self.proj_in(x) 
        x = self.conv(x)

        # -- spectral pooling 
        chunks = [feat for feat in torch.chunk(x.clone(), self.n_chunk, dim=1)]
        feat_bank = [self.sps[i](chunks[i]) for i in range(self.n_chunk)] 
        self.ctx = self.pws[0](feat_bank[0]) + self.pws[1](feat_bank[1]) + self.pws[2](feat_bank[2]) # context aggregation
        x = x * self.ctx # modulation

        # == Update == # 
        x = self.proj_out(x) 
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        """ x : (B, C, H, W) 
        """
        return x * self.scale.unsqueeze(-1).unsqueeze(-1)



class SPANetBlock(nn.Module):
    """
    Implementation of one SPANet block.
    --dim: embedding dim
    --k_size: kernel size 
    --patch_dim: patch size 
    --r: radius of filter 
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, k_size=7, patch_dim=224//4, r=2**1, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., res_scale_init_value=None):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = SPAM(dim=dim, k_size=k_size, H=patch_dim,W=patch_dim, r=r)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep SPANets.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()


    def forward(self, x):
        x = self.res_scale1(x) + self.drop_path(self.token_mixer(self.norm1(x)))
        x = self.res_scale2(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers, 
                 k_size=7, patch_dim=224//4, r=2**1, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., 
                 res_scale_init_value=None):
    """
    generate SPANet blocks for a stage
    return: SPANet blocks 
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(SPANetBlock(
            dim, k_size=k_size, patch_dim=patch_dim, r=r, mlp_ratio=mlp_ratio, 
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            res_scale_init_value=res_scale_init_value, 
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class SPANet(nn.Module):
    """
    SPANet, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --k_size: the embedding dims, mlp ratios and 
        pooling size for the 4 stages
    --patch_dims: the patch size for the 4 stages
    --radius(list): radius of filter; [2**4, 2**3, 2**2, 2**1]
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad: 
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained: 
        for mmdetection and mmsegmentation to load pretrained weights
    """
    def __init__(self, layers, embed_dims=None, patch_dims=None,
                 mlp_ratios=None, downsamples=None, 
                 k_size=7, 
                 radius=[2**1, 2**1, 2**0, 2**0],
                 norm_layer=GroupNorm, act_layer=nn.GELU, 
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2, 
                 down_patch_size=3, down_stride=2, down_pad=1, 
                 drop_rate=0., drop_path_rate=0.,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 fork_feat=False,
                 init_cfg=None, 
                 pretrained=None, 
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, 
                                 k_size=k_size, patch_dim= patch_dims[i], r=radius[i], mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 res_scale_init_value=res_scale_init_values[i])
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride, 
                        padding=down_pad, 
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model 
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading 
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            
            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out



model_urls = {
    "spanet-small": None,
    "spanet-medium": None,
    "spanet-base": None,
    
}


@register_model
def spanet_small(pretrained=False, **kwargs):
    """
    SPANet-small model,
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    img_size = 224
    patch_dims = [img_size//2**2, img_size//2**3, img_size//2**4, img_size//2**5]    
    radius=[2**1, 2**1, 2**0, 2**0]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = SPANet(
        layers, embed_dims=embed_dims, patch_dims = patch_dims, radius=radius, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['spanet_s']
    if pretrained:
        url = model_urls['spanet-small']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model



@register_model
def spanet_medium(pretrained=False, **kwargs):
    """
    SPANet-medium model, 
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 320, 512]
    img_size = 224
    patch_dims = [img_size//2**2, img_size//2**3, img_size//2**4, img_size//2**5]
    radius=[2**1, 2**1, 2**0, 2**0]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = SPANet(
        layers, embed_dims=embed_dims, patch_dims = patch_dims, radius=radius,
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['spanet_m']
    if pretrained:
        url = model_urls['spanet-medium']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def spanet_mediumX(pretrained=False, **kwargs):
    """
    SPANet-mediumX model, 
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [8, 8, 24, 8]
    embed_dims = [64, 128, 320, 512]
    img_size = 224
    patch_dims = [img_size//2**2, img_size//2**3, img_size//2**4, img_size//2**5]
    radius=[2**1, 2**1, 2**0, 2**0]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = SPANet(
        layers, embed_dims=embed_dims, patch_dims = patch_dims, radius=radius,
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['spanet_m']
    if pretrained:
        url = model_urls['spanet-medium']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def spanet_base(pretrained=False, **kwargs):
    """
    SPANet-base model, 
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [6, 6, 18, 6]
    embed_dims = [96, 192, 384, 768]
    img_size = 224
    patch_dims = [img_size//2**2, img_size//2**3, img_size//2**4, img_size//2**5]
    radius=[2**1, 2**1, 2**0, 2**0]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = SPANet(
        layers, embed_dims=embed_dims, patch_dims = patch_dims, radius=radius,
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['spanet_b']
    if pretrained:
        url = model_urls['spanet-base']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def spanet_baseX(pretrained=False, **kwargs):
    """
    SPANet-base model, 
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [8, 8, 24, 8]
    embed_dims = [96, 192, 384, 768]
    img_size = 224
    patch_dims = [img_size//2**2, img_size//2**3, img_size//2**4, img_size//2**5]
    radius=[2**1, 2**1, 2**0, 2**0]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = SPANet(
        layers, embed_dims=embed_dims, patch_dims = patch_dims, radius=radius,
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['spanet_b']
    if pretrained:
        url = model_urls['spanet-base']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model



if __name__ == "__main__": 
    import torch

    model = spanet_small()
    model.eval()

    image_size = [224, 224]
    input = torch.rand(1, 3, *image_size)

    output = model(input)
    print(output.shape)


