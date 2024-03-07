"""
Sourced from official implementation of CvT, see:
https://github.com/microsoft/CvT/blob/main/lib/models/cls_cvt.py
"""
import os
import logging

import numpy as np
import scipy
import torch
import torch.nn as nn

from functools import partial

from .layers import VisionTransformer, QuickGELU

# patch stride changed to 2,2,2 to reduce upsampling burden
# patch padding increased to handle edge cases
DEFAULT_SPEC = {
    'INIT': 'trunc_norm',
    'NUM_STAGES': 3,
    'PATCH_SIZE': [7, 3, 3],
    'PATCH_STRIDE': [2, 2, 2],
    'PATCH_PADDING': [3, 1, 1],
    'DIM_EMBED': [64, 192, 384],
    # 'DIM_EMBED': [32, 96, 192],
    'NUM_HEADS': [1, 3, 6],
    'DEPTH': [1, 3, 14],
    'MLP_RATIO': [4.0, 4.0, 4.0],
    'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
    'DROP_RATE': [0.0, 0.0, 0.0],
    'DROP_PATH_RATE': [0.0, 0.0, 0.1],
    'QKV_BIAS': [True, True, True],
    'CLS_TOKEN': [False, False, False],
    'POS_EMBED': [False, False, False],
    'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
    'KERNEL_QKV': [3, 3, 3],
    'PADDING_KV': [1, 1, 1],
    'STRIDE_KV': [2, 2, 2],
    'PADDING_Q': [1, 1, 1],
    'STRIDE_Q': [1, 1, 1],
}

class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 bottleneck_features=1024,
                 act_layer=QuickGELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-5),
                 init='trunc_norm',
                 dp_cross_attention=False,
                 spec=DEFAULT_SPEC):
        
        super().__init__()
        # self.num_classes = num_classes
        # print(f'Cross attention mode: {dp_cross_attention}')
        self.num_stages = spec['NUM_STAGES']

        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                dp_cross_attention=dp_cross_attention,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]
        
        dim_embed = spec['DIM_EMBED'][-1]
        self.expander = nn.Conv2d(dim_embed,
                                  bottleneck_features,
                                  kernel_size=[2,2],
                                  stride=[2,2],
                                  padding=[1,1])
        
        self.expander_norm = nn.BatchNorm2d(bottleneck_features)
        self.expander_act = nn.LeakyReLU()

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    if 'pos_embed' in k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()
                        logging.info(
                            '=> load_pretrained: resized variant: {} to {}'
                            .format(size_pretrained, size_new)
                        )

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))

                        logging.info(
                            '=> load_pretrained: grid-size from {} to {}'
                            .format(gs_old, gs_new)
                        )

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = scipy.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = torch.tensor(
                            np.concatenate([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f'stage{i}.pos_embed')
            layers.add(f'stage{i}.cls_token')

        return layers

    def forward_features(self, x):
        features = []
        for i in range(self.num_stages):
            x, _ = getattr(self, f'stage{i}')(x)
            features.append(x)
            
        return features
    
    def forward(self, x):
        features = self.forward_features(x)
        x = self.expander(features[-1])
        x = self.expander_act(self.expander_norm(x)) # NOTE 
        return x, features