import torch
from torch import nn as nn
from functools import partial

from .base_block import ConvBlock
from .ops import FeaturesConnector


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, activation, pool, padding):
        super(UNetDownBlock, self).__init__()
        self.convs = UNetDoubleConv(
            in_channels, out_channels,
            norm_layer=norm_layer, activation=activation, padding=padding,
        )
        self.pooling = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        conv_x = self.convs(x)
        return conv_x, self.pooling(conv_x)


class UNetUpBlock(nn.Module):
    def __init__(
        self,
        in_channels_decoder, in_channels_encoder, out_channels,
        norm_layer, activation, padding,
        attention_layer,
    ):
        super(UNetUpBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(
                in_channels_decoder, out_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=None, activation=activation,
            )
        )
        self.convs = UNetDoubleConv(
            in_channels_encoder + out_channels, out_channels,
            norm_layer=norm_layer, activation=activation, padding=padding,
        )
        if attention_layer is not None:
            self.attention = attention_layer(in_channels_encoder + out_channels, norm_layer, activation)
        else:
            self.attention = None

    def forward(self, x, encoder_out, mask=None):
        upsample_x = self.upconv(x)
        x_cat_encoder = torch.cat([encoder_out, upsample_x], dim=1)
        if self.attention is not None:
            x_cat_encoder = self.attention(x_cat_encoder, mask)
        return self.convs(x_cat_encoder)


class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, activation, padding):
        super(UNetDoubleConv, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=padding,
                norm_layer=norm_layer, activation=activation,
            ),
            ConvBlock(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=padding,
                norm_layer=norm_layer, activation=activation,
            ),
        )

    def forward(self, x):
        return self.block(x)