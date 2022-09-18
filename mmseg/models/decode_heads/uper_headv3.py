# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize

from .lib.rcfpn import RCFPN
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from .lib.attention import *
from .lib.bifpn import BiFPN
from mmseg.models.utils.se_layer import *
from torch import nn
from .lib.mlp_osa import MLP_OSA


"""
 *: experimenting
 X: not good
 V: good
 ~: hesitating
  : just idea
"""

# BAM module instead of RA          X

@HEADS.register_module()
class UPerHeadV3(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHeadV3, self).__init__(
            input_transform='multiple_select', **kwargs)

    

        # self.fuse_feature = BiFPN(self.in_channels, self.channels)
        self.fuse_feature = RCFPN(self.in_channels, self.channels, 4)
        
        
        self.mlp_slow = MLP_OSA(in_channels=self.in_channels, channels=self.channels)

        
        self.layer_attn = LayerAttention(
            self.channels,
            groups=len(self.in_channels), la_down_rate=8
        )
        
        self.reverse_attn = ReverseAttention(
            self.channels,
            1,
            self.conv_cfg, self.norm_cfg, self.act_cfg
        )
        # self.bam = BAM(in_channels=self.channels)

    
    def forward(self, inputs):

        inputs = self._transform_inputs(inputs)
        # inputs.append(self.psp_forward(inputs))

        # build top-down path 3, 2, 1
        fpn_outs = self.fuse_feature(inputs)
      
        out = self.mlp_slow(fpn_outs)

        
        ## edge attention
        out = self.reverse_attn(out)


        output = self.cls_seg(out)
        return output


