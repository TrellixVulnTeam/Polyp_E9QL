import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class MLPSLowHead(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 ops='cat',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        assert ops in ['cat', 'add']
        self.ops = ops
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.linear_projections = nn.ModuleList()
        for i in range(num_inputs - 1):
            self.linear_projections.append(
                ConvModule(
                    in_channels=self.channels * 2 if self.ops == 'cat' else self.channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        _inputs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        out = torch.empty(
            _inputs[0].shape
        )
        for idx in range(len(_inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = out
                x2 = _inputs[idx - 1]
            if self.ops == 'cat':
                x = torch.cat([x1, x2], dim=1)
            else:
                x = x1 + x2
            out = linear_prj(x)

        out = self.cls_seg(out)

        return out


@HEADS.register_module()
class MLPSLowHead_v2(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 ops='cat',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        assert ops in ['cat', 'add']
        self.ops = ops
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.linear_projections = nn.ModuleList()
        for i in range(num_inputs - 1):
            self.linear_projections.append(
                ConvModule(
                    in_channels=self.channels * 2 if self.ops == 'cat' else self.channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        _inputs = [] # 1/4, 1/8, 1/16, 1/32
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(conv(x))

        # slow concatenate
        out = torch.empty(
            _inputs[0].shape
        )
        for idx in range(len(_inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = out
                x2 = _inputs[idx - 1]
            if self.ops == 'cat':
                x1 = resize(
                    input=x1,
                    size=x2.shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
                x = torch.cat([x1, x2], dim=1)
            else:
                x1 = resize(
                    input=x1,
                    size=x2.shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
                x = x1 + x2
            out = linear_prj(x)

        out = self.cls_seg(out)

        return out