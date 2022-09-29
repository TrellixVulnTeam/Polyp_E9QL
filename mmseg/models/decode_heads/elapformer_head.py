import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.utils import SELayer


@HEADS.register_module()
class ELAPFormerHead(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 **kwargs):
        super(ELAPFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # feature fusion between adjacent levels
        self.linear_projections = nn.ModuleList()
        for i in range(num_inputs - 1):
            self.linear_projections.append(
                ConvModule(
                    in_channels=self.channels * 2,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.sep_conv = nn.ModuleList([
            nn.Conv2d(self.channels, self.channels, 1),
            nn.Conv2d(self.channels, self.channels, 1)
        ])

        self.se_module = SELayer(
            channels=self.channels * (num_inputs)
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels * (num_inputs),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        _inputs = [] # 1/4, 1/8, 1/8, 1/8
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            feat = conv(x)
            if idx > 0:
                feat = resize(
                    input=feat,
                    size=inputs[1].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            _inputs.append(feat)

        # progressive feature fusion
        # _out to cache features
        _out = torch.empty(
            _inputs[1].shape
        )
        outs = [_inputs[-1]] # 1/8, 1/8, 1/8, 1/8, 1/4
        for idx in range(len(inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            # upsampling the 2nd scale to size of 1st scale to concat
            if idx == 1:
                x1 = resize(
                    input=x1,
                    size=x2.shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            # decoupling the 1/4 scale in outs
            if idx == 1:
                _out_sep1 = self.sep_conv[0](_out)
                _out_sep2 = self.sep_conv[1](_out)
                # resize half of 1/4 to 1/8
                _out_sep1 = resize(
                    input=_out_sep1,
                    size=inputs[1].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
                outs.extend([_out_sep1, _out_sep2])
            else: # if not 1/4 then append to list
                outs.append(_out)

        out = torch.cat(outs[:4], dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)
        # upsample all 1/8 to 1/4
        out = resize(
            input=out,
            size=outs[-1].shape[2:],
            mode=self.interpolate_mode,
            align_corners=self.align_corners
        )
        # perform identity mapping
        out = outs[-1] + out

        out = self.cls_seg(out)

        return out


@HEADS.register_module()
class ELAPFormerHead_v2(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 **kwargs):
        super(ELAPFormerHead_v2, self).__init__(input_transform='multiple_select', **kwargs)
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # feature fusion between adjacent levels
        self.linear_projections = nn.ModuleList()
        for i in range(num_inputs - 1):
            self.linear_projections.append(
                ConvModule(
                    in_channels=self.channels * 2,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.sep_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.channels, self.channels, 1),
                nn.Conv2d(self.channels,
                          self.channels,
                          kernel_size=2,
                          stride=2)
            ),
            nn.Conv2d(self.channels, self.channels, 1)
        ])

        self.se_module = SELayer(
            channels=self.channels * (num_inputs)
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels * (num_inputs),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        _inputs = [] # 1/4, 1/8, 1/8, 1/8
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            feat = conv(x)
            if idx > 0:
                feat = resize(
                    input=feat,
                    size=inputs[1].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            _inputs.append(feat)

        # progressive feature fusion
        # _out to cache features
        _out = torch.empty(
            _inputs[1].shape
        )
        outs = [_inputs[-1]] # 1/8, 1/8, 1/8, 1/8, 1/4
        for idx in range(len(inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            # upsampling the 2nd scale to size of 1st scale to concat
            if idx == 1:
                x1 = resize(
                    input=x1,
                    size=x2.shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            # decoupling the 1/4 scale in outs
            if idx == 1:
                _out_sep1 = self.sep_conv[0](_out)
                _out_sep2 = self.sep_conv[1](_out)
                # resize half of 1/4 to 1/8
                _out_sep1 = resize(
                    input=_out_sep1,
                    size=inputs[1].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
                outs.extend([_out_sep1, _out_sep2])
            else: # if not 1/4 then append to list
                outs.append(_out)

        out = torch.cat(outs[:4], dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)
        # upsample all 1/8 to 1/4
        out = resize(
            input=out,
            size=outs[-1].shape[2:],
            mode=self.interpolate_mode,
            align_corners=self.align_corners
        )
        # perform identity mapping
        out = outs[-1] + out

        out = self.cls_seg(out)

        return out