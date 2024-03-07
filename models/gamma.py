import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .miniViT import mViT
# from transformers import CvtConfig, CvtForImageClassification
from .cvt import ConvolutionalVisionTransformer, DEFAULT_SPEC
from .convlstm import ConvLSTM
from .decoder import DecoderCVT

class Gamma(nn.Module):
    def __init__(self,
                 in_chans=10,
                 layers=3,
                 rank=12,
                 bottleneck_features=1024,
                 temporal=False,
                 dp_cross_attention=False,
                 cvt_spec=DEFAULT_SPEC):

        super(Gamma, self).__init__()
        self.layers = layers
        self.rank = rank
        td_chans = layers*rank*3
        self.dp_cross_attention = dp_cross_attention 
        
        if self.dp_cross_attention:
            raise NotImplementedError('Not implemented yet.')
        
        self.cvt_embed_dims = cvt_spec['DIM_EMBED']

        self.encoder = ConvolutionalVisionTransformer(
            in_chans=in_chans,
            spec=cvt_spec,
            bottleneck_features=bottleneck_features,
            dp_cross_attention=dp_cross_attention,
        )
        self.temporal = temporal

        if temporal:
            self.state_model = ConvLSTM(input_size=bottleneck_features,
                                        hidden_size=bottleneck_features,
                                        kernel_size=3)
            
        self.decoder = DecoderCVT(num_classes=td_chans,
                                  feature_depths=cvt_spec['DIM_EMBED'],
                                  bottleneck_features=bottleneck_features,
                                  )
        
        self.adaptive_bins_layer = mViT(in_chans,
                                        patch_size=16,
                                        dim_out=layers,
                                        embedding_dim=128)
        


    def forward(self, x, prev_state=None, **kwargs):
        # Input format: batch_size, in_chans, height, width
        x_new, all_outs = self.encoder(x)
        if self.temporal:
            state = self.state_model(x_new, prev_state)
            x_new=state[1]
        else:
            state=None
        x_new = self.decoder(x_new, all_outs)
        out = rearrange(x_new,
                        'b (l r c) h w -> b l r c h w',
                        l=self.layers, r=self.rank, c=3)

        depth_planes = self.adaptive_bins_layer(x)
        # print(depth_planes.shape, "\nDepth Plane out:", depth_planes)
        return out, depth_planes, state

    @classmethod
    def build(cls, **kwargs):
        m = cls(**kwargs)
        print('Initialized Gamma.')
        return m