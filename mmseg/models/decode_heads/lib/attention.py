# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:15:44 2021

@author: angelou
"""

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv
import math
from mmcv.cnn import ConvModule
from mmseg.models.utils import *
class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1),stride=1,padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)
        projected_query = self.query_conv(x).reshape(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).reshape(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).reshape(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out
    




class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=5, dilation = 2):
        super(SpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        
        self.in_channels = in_channels
        pad_1 = (self.kernel_size-1)//2  # Padding on both side for dilation 1
        pad_2 = (2*pad_1 + (self.kernel_size-1)*(dilation-1))//2 # Padding on both side for dilation n
        pad_3 = (2*pad_1 + (self.kernel_size-1)*(dilation-2))//2 # Padding on both side for dilation n-1

        self.grp1_conv1k = nn.Conv2d(self.in_channels, self.in_channels//2, self.kernel_size, padding=pad_2, dilation=dilation)
        self.grp1_bn1 = nn.BatchNorm2d(self.in_channels//2)
        self.grp1_convk1 = nn.Conv2d(self.in_channels//2, 1, self.kernel_size, padding=pad_3, dilation=dilation-1)
        self.grp1_bn2 = nn.BatchNorm2d(1)


    def forward(self, input_):
        # Generate Group 1 Features
        grp1_feats = self.grp1_conv1k(input_)
        grp1_feats = F.relu(self.grp1_bn1(grp1_feats))
        grp1_feats = self.grp1_convk1(grp1_feats)
        grp1_feats = F.relu(self.grp1_bn2(grp1_feats))


        added_feats = torch.sigmoid(grp1_feats)
        added_feats = added_feats.expand_as(input_).clone()

        return added_feats


class ChannelwiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelwiseAttention, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels//4)
        self.linear_2 = nn.Linear(self.in_channels//4, self.in_channels)

    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))
        
        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()

        return feats


class LayerAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 groups, la_down_rate=8):
        super(LayerAttention, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.groups,
                kernel_size=3, padding=1
            ),
            nn.Sigmoid()
        )
        
        # self.la_conv = ConvModule(self.in_channels, self.in_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)


    def forward(self, x):
        b, c, h, w = x.shape

        avg_feat = nn.AdaptiveAvgPool2d(1)(x)           # average pooling like every fucking attention do
        weight = self.layer_attention(avg_feat)         # make weight of shape (b, groups, 1, 1)

        x = x.view(b, self.groups, c // self.groups, h, w)
        weight = weight.view(b, self.groups, 1, 1, 1)
        _x = x.clone()
        for group in range(self.groups):
            _x[:, group] = x[:, group] * weight[:, group]

        _x = _x.view(b, c, h, w)
        # _x = self.la_conv(_x)

        return _x


        
class ReverseAttention(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.fpn_bottleneck = ConvModule(
            self.in_channels, self.out_channels,
            kernel_size=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.ra_conv = ConvModule(
            self.in_channels , self.in_channels, kernel_size=3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
    def forward(self, input):
        
        out = self.fpn_bottleneck(input)
        out = -1*(torch.sigmoid(out)) + 1
        out = out.expand(-1, self.in_channels, -1, -1).mul(input)
        out = self.ra_conv(out)
        return out
    


class EfficientSELayer(nn.Module):
    def __init__(self,
                 channels,
                 conv_cfg=None):
        super(EfficientSELayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.attn_weight = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                conv_cfg=conv_cfg,
                act_cfg=None
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.attn_weight(out)

        return x * out


class GatingSignal(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, input, skip):
        signal = F.relu(self.conv(input))
        signal = self.bn(signal)
        return signal

class Gamma(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(
                torch.ones((1,1), dtype=torch.float32), requires_grad=True)
    def forward(self, input):
        input = torch.clamp(input, 1e-7, 1.)
        return input**self.w

class FocusGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attn = ChannelwiseAttention(channels)
        self.spatial_attn = SpatialAttention(channels)
        self.gamma = Gamma()
        self.out_conv = nn.Conv2d(channels, channels,kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
    def forward(self, input):
        channel_attn = self.channel_attn(input)
        spatial_attn = self.spatial_attn(input)
        weights = torch.multiply(channel_attn, spatial_attn)
        weights = self.gamma(weights)
        output = torch.multiply(weights, input)
        # output = self.bn(self.out_conv(weights))
        return output


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se1=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        
        self.se2=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se1(max_result)
        avg_out=self.se2(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output

from torch.nn import init

class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual


class ParallelPolarizedSelfAttention(nn.Module):
    
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        out=spatial_out+channel_out
        return out






class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*channel_out
        return spatial_out

class BAM(nn.Module):
    def __init__(self,in_channels):
        super(BAM, self).__init__()
        
        self.boundary_conv=nn.Sequential(
            nn.Conv2d(in_channels, in_channels//3,3,1,1),
            nn.BatchNorm2d(in_channels//3),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.foregound_conv=nn.Sequential(
            nn.Conv2d(in_channels, in_channels//3,3,1, 1),
            nn.BatchNorm2d(in_channels//3),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.background_conv=nn.Sequential(
            nn.Conv2d(in_channels, in_channels//3,3,1, 1),
            nn.BatchNorm2d(in_channels//3),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.out_conv=nn.Sequential(
            nn.Conv2d((in_channels//3)*3, in_channels,3,1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d())
        self.fpn_bottleneck = ConvModule(
            in_channels, 1 ,kernel_size=1, padding=0)
        self.selayer=SELayer((in_channels//3)*3)

    def forward(self, x):
        residual = x
        pred = self.fpn_bottleneck(x)
        score = torch.sigmoid(pred)
        
        #boundary
        dist = torch.abs(score - 0.5)
        boundary_att = 1 - (dist / 0.5)
        boundary_x = x * boundary_att
        
        #foregound
        foregound_att= score
        foregound_att=torch.clip(foregound_att-boundary_att,0,1)
        foregound_x= x*foregound_att

        #background
        background_att=1-score
        background_att=torch.clip(background_att-boundary_att,0,1)
        background_x= x*background_att

        foregound_x= foregound_x 
        background_x= background_x 
        boundary_x= boundary_x  

        foregound_xx=self.foregound_conv(foregound_x)
        background_xx=self.background_conv(background_x)
        boundary_xx=self.boundary_conv(boundary_x)

        out=torch.cat([foregound_xx,background_xx,boundary_xx], dim=1) 
        out=self.selayer(out)
        out=self.out_conv(out)+residual
        return out
    
test1 = torch.rand((2, 64, 128, 128))
test2 = torch.rand((2, 64, 128, 128))

fg = BAM(64)
print(fg(test1).shape)
        
        
        

