# Copyright (c) OpenMMLab. All rights reserved.

# this module is used only as a perceptual loss

import warnings
import torch

from torch import nn


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.
    If style is "pytorch", the stride-two layer is the 3x3 conv layer, if it is
    "caffe", the stride-two layer is the first 1x1 conv layer.
    """

    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int=1,
                 dilation: int=1,
                 downsample: nn.Module=None):
        super(Bottleneck, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation

        self.conv1_stride = 1
        self.conv2_stride = stride

        self.norm1_name, norm1 = 'bn1', nn.BatchNorm2d(planes)
        self.norm2_name, norm2 = 'bn2', nn.BatchNorm2d(planes)
        self.norm3_name, norm3 = 'bn3', nn.BatchNorm2d(planes * self.expansion)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False
        )

        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = self.relu(_inner_forward(x))

        return out


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.
    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        multi_grid (int | None): Multi grid dilation rates of last
            stage. Default: None
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False
    """

    def __init__(self,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 dilation=1,
                 avg_down=False,
                 contract_dilation=False):
        downsample = None
        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                nn.Conv2d(
                    inplanes,
                    planes * Bottleneck.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion)
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if dilation > 1 and contract_dilation:
                first_dilation = dilation // 2
        else:
                first_dilation = dilation
        layers.append(
            Bottleneck(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=first_dilation,
                downsample=downsample))
        inplanes = planes * Bottleneck.expansion
        for i in range(1, num_blocks):
            layers.append(
                Bottleneck(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation))
        super(ResLayer, self).__init__(*layers)
