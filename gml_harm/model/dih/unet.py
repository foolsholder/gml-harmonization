import torch
from torch import nn as nn
from functools import partial

from gml_harm.core.layer.base_block import ConvBlock
from gml_harm.core.layer.ops import FeaturesConnector
from gml_harm.core.layer.unet_block import *


class UNetEncoder(nn.Module):
    def __init__(
        self,
        depth, out_channels, in_channels,
        norm_layer, batchnorm_from, max_channels,
        backbone_from=0, backbone_channels=None, backbone_mode=''
    ):
        super(UNetEncoder, self).__init__()
        self.depth = depth
        self.backbone_from = backbone_from
        self.block_channels = []
        backbone_channels = [] if backbone_channels is None else backbone_channels[::-1]
        relu = partial(nn.ReLU, inplace=True)

        possible_normlayes = {
            "BatchNorm2d": nn.BatchNorm2d
        }
        norm_layer = possible_normlayes[norm_layer]

        self.block0 = UNetDownBlock(
            in_channels, out_channels,
            norm_layer=norm_layer if batchnorm_from == 0 else None,
            activation=relu,
            pool=True, padding=1,
        )
        self.block_channels.append(out_channels)
        in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
        self.block1 = UNetDownBlock(
            in_channels, out_channels,
            norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None,
            activation=relu,
            pool=True, padding=1,
        )
        self.block_channels.append(out_channels)

        self.blocks_connected = nn.ModuleDict()
        self.connectors = nn.ModuleDict()
        for block_i in range(2, depth):
            in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
            if 0 <= backbone_from <= block_i and len(backbone_channels):
                stage_channels = backbone_channels.pop()
                connector = FeaturesConnector(backbone_mode, in_channels, stage_channels, in_channels)
                self.connectors[f'connector{block_i}'] = connector
                in_channels = connector.output_channels
            self.blocks_connected[f'block{block_i}'] = UNetDownBlock(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                activation=relu, padding=1,
                pool=block_i < depth - 1,
            )
            self.block_channels.append(out_channels)

    def forward(self, x, backbone_features):
        backbone_features = [] if backbone_features is None else backbone_features[::-1]
        outputs = []

        block_input = x
        output, block_input = self.block0(block_input)
        outputs.append(output)
        output, block_input = self.block1(block_input)
        outputs.append(output)

        for block_i in range(2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            connector_name = f'connector{block_i}'
            if connector_name in self.connectors:
                stage_features = backbone_features.pop()
                connector = self.connectors[connector_name]
                block_input = connector(block_input, stage_features)
            output, block_input = block(block_input)
            outputs.append(output)

        return outputs[::-1]


class UNetDecoder(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer,
                 attention_layer=None, attend_from=3, image_fusion=False):
        super(UNetDecoder, self).__init__()
        self.up_blocks = nn.ModuleList()
        self.image_fusion = image_fusion
        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        # Last encoder layer doesn't pool, so there're only (depth - 1) deconvs

        possible_normlayes = {
            "BatchNorm2d": nn.BatchNorm2d
        }
        norm_layer = possible_normlayes[norm_layer]

        for d in range(depth - 1):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            stage_attention_layer = attention_layer if 0 <= attend_from <= d else None
            self.up_blocks.append(UNetUpBlock(
                in_channels, out_channels, out_channels,
                norm_layer=norm_layer, activation=partial(nn.ReLU, inplace=True),
                padding=1,
                attention_layer=stage_attention_layer,
            ))
            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, encoder_outputs, input_image, mask):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.up_blocks, encoder_outputs[1:]):
            output = block(output, skip_output, mask)

        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output = attention_map * input_image + (1.0 - attention_map) * self.to_rgb(output)
        else:
            output = self.to_rgb(output)

        return output


class UNet(nn.Module):
    def __init__(self, model_cfg):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(**model_cfg['encoder'])
        self.decoder = UNetDecoder(encoder_blocks_channels=self.encoder.block_channels,
                                   **model_cfg['decoder'])

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x = torch.cat([images, masks], dim=1)
        encoder_feats = self.encoder(x, backbone_features=None)
        outputs = self.decoder(encoder_feats, images, masks)
        return outputs