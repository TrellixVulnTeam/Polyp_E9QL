import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from .attention import *
from .context_module import CFPModule
from mmseg.ops import resize
from mmseg.models.utils import SELayer
from .axial_attention import AA_kernel 
from .attention import FocusGate
class MLP_OSA(nn.Module):
    def __init__(self,
                 interpolate_mode='bilinear',
                 ops='cat', in_channels=None, channels=None,
                 **kwargs):
        super().__init__()

        self.interpolate_mode = interpolate_mode
        assert ops in ['cat', 'add']
        self.ops = ops
        self.in_channels = in_channels
        self.channels = channels
        num_inputs = len(self.in_channels)


        self.linear_projections = nn.ModuleList()
        for i in range(num_inputs - 1):
            self.linear_projections.append(
                nn.Sequential(
                    ConvModule(
                    in_channels=self.channels * 2 if self.ops == 'cat' else self.channels,
                    out_channels=self.channels, norm_cfg=None,
                    kernel_size=1, 
                    padding=0),
                )
            )

        self.aa_module = AA_kernel(self.channels, self.channels)
        self.CFP_1 = CFPModule(self.channels, d = 8)
        self.fpn_bottleneck = ConvModule(self.channels, 1,kernel_size=1, padding=0)
        
        self.ra_conv = ConvModule(self.channels , self.channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        _inputs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            _inputs.append(
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=False))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = [_inputs[-1]]
        for idx in range(len(_inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            if self.ops == 'cat':
                x = torch.cat([x1, x2], dim=1)
            else:
                x = x1 + x2
            _out = linear_prj(x)
            outs.append(_out)
            
        # out = torch.cat(outs, dim=1)
        # out = self.layer_attn(out)
        # out = self.fusion_conv(out)
        ra_atten = self.fpn_bottleneck(_out)
        ra_atten = -1*(torch.sigmoid(ra_atten)) + 1
        
        cfp_out = self.CFP_1(_out) 
        aa_atten = self.aa_module(cfp_out)
        aa_atten += cfp_out + outs[-1]
        
        out = ra_atten.expand(-1, self.channels, -1, -1).mul(aa_atten)
        out = self.ra_conv(out)
        
        return out
    


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from mmcv.cnn import ConvModule, xavier_init, constant_init
from .lib.attention import SequentialPolarizedSelfAttention, CBAMBlock
from .psp_head import PPM
from mmseg.ops import resize



# Fusion node 3 -> 2
# upsample attention and psa softmax -> sigmoid
class FusionNode(nn.Module):

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 with_out_conv=True,
                 out_conv_cfg=None,
                 out_norm_cfg=None,
                 upsample_mode='bilinear',
                 op_num=2,
                 upsample_attn=False):
        super(FusionNode, self).__init__()
        assert op_num == 2 or op_num == 3
        self.with_out_conv = with_out_conv
        self.upsample_mode = upsample_mode
        self.op_num = op_num
        self.upsample_attn = upsample_attn
        act_cfg = None
        self.act_cfg = act_cfg

        self.weight = nn.ModuleList()
        self.gap = nn.AdaptiveAvgPool2d(1)
        for i in range(op_num - 1):
            self.weight.append(
                nn.Conv2d(in_channels * 2, 1, kernel_size=1, bias=True))
            constant_init(self.weight[-1], 0)

        if self.upsample_attn:
            self.spatial_weight = nn.Conv2d(
                in_channels * 2, 1, kernel_size=3, padding=1, bias=True)
            self.temp = nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=True)
            for m in self.spatial_weight.modules():
                if isinstance(m, nn.Conv2d):
                    constant_init(m, 0)

        if self.with_out_conv:
            self.post_fusion = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=out_conv_cfg,
                norm_cfg=out_norm_cfg,
                order=('act', 'conv', 'norm'))
            self.gate = SequentialPolarizedSelfAttention(channel=out_channels)
            if out_conv_cfg is None or out_conv_cfg['type'] == 'Conv2d':
                for m in self.post_fusion.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform')

        if op_num > 2:
            self.pre_fusion = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=out_conv_cfg,
                norm_cfg=out_norm_cfg,
                order=('act', 'conv', 'norm'))
            if out_conv_cfg is None or out_conv_cfg['type'] == 'Conv2d':
                for m in self.pre_fusion.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform')

    def dynamicFusion(self, x):
        x1, x2 = x[0], x[1]
        
        batch, channel, height, width = x1.size()
        weight1 = self.gap(x1)
        weight2 = self.gap(x2)
        if self.upsample_attn:
            upsample_weight = (
                self.temp * channel**(-0.5) *
                self.spatial_weight(torch.cat((x1, x2), dim=1)))
            upsample_weight = F.softmax(
                upsample_weight.reshape(batch, 1, -1), dim=-1).reshape(
                    batch, 1, height, width) * height * width
            x2 = upsample_weight * x2
        weight = torch.cat((weight1, weight2), dim=1)
        weight = self.weight[0](weight)
        weight = torch.sigmoid(weight)
        result = weight * x1 + (1 - weight) * x2
        if self.op_num == 3:
            x3 = x[2]
            x1 = self.pre_fusion(result)
            # x1 = result
            
            weight1 = self.gap(x1)
            weight3 = self.gap(x3)
            weight = torch.cat((weight1, weight3), dim=1)
            weight = self.weight[1](weight)
            weight = torch.sigmoid(weight)
            result = weight * x1 + (1 - weight) * x3
        if self.with_out_conv:
            result = self.post_fusion(result)
            result = self.gate(result)
        return result

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] < size:
            return F.interpolate(x, size=size, mode=self.upsample_mode)
        else:
            _, _, h, w = x.size()
            x = F.max_pool2d(
                F.pad(x, [0, w % 2, 0, h % 2], 'replicate'), (2, 2))
            return x

    def forward(self, x):
        out_size=x[0].shape[-2:]
        inputs = []
        
        for feat in x:
            inputs.append(self._resize(feat, out_size))

            
        return self.dynamicFusion(inputs)


class RCFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 pool_scales=(1, 2, 3, 6)):
        super(RCFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)  # num of input feature levels
        self.num_outs = num_outs  # num of output feature levels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.out_channels,
            conv_cfg=None,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=False)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.out_channels,
            self.out_channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # add lateral connections
        self.lateral_convs = nn.ModuleList()
        # self.gates = nn.ModuleList()
        self.RevFP = nn.ModuleList()
        
        for i in range(self.start_level, self.backbone_end_level - 1):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1, padding=0,   
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.lateral_convs.append(l_conv)
            rev_fuse = FusionNode(
            in_channels=out_channels,
            out_channels=out_channels,
            out_conv_cfg=None,
            out_norm_cfg=norm_cfg,
            op_num=3)
            self.RevFP.append(rev_fuse)
        
            
    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.lateral_convs.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output
    def forward(self, inputs):
        """Forward function."""
        # build P3-P5
        feats = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        feats.append(self.psp_forward(inputs))
        
        used_backbone_levels = len(feats)
        for i in range(used_backbone_levels - 1, 0, -1):
            feats[i-1] = self.RevFP[i-1]([feats[i-1], feats[i], feats[i-1]])


        return feats