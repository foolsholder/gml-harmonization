import torch

from copy import copy
from torch import nn
from typing import Union, Tuple, Optional, List


class ConvNormAct(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, ...]] = 3,
            padding: Union[int, Tuple[int, ...]] = 1,
            stride: Union[int, Tuple[int, ...]] = 1,
            norm_layer: Optional[nn.Module] = nn.BatchNorm2d,
            activation: nn.Module = nn.ELU,
            conv_layer: nn.Module = nn.Conv2d,
            masked_norm: bool = False
    ):
        super(ConvNormAct, self).__init__()
        self.conv = conv_layer(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        self.act = activation(inplace=True)
        self.masked_norm = masked_norm

    def forward(self, x, mask=None):
        if self.masked_norm:
            return self.act(self.norm(self.conv(x), mask))
        return self.act(self.norm(self.conv(x)))


class FeatureAggregation(nn.Module):
    def __init__(
            self,
            a_channels: int,
            b_channels: int,
            mode: str = 'cat',
            out_channels: int = None
    ):
        super(FeatureAggregation, self).__init__()
        """
        mode \in {'cat', 'sum', 'conv'}
        if 'conv' then out_channels must be provided
        if 'sum' then a_channels must be equal to b_channels
        """
        self.mode = mode

        if mode == 'cat':
            self.out_channels = a_channels + b_channels
        else:
            raise "Not implemented yet - feature connector with mode={}".format(mode)

    def forward(self, fst: torch.Tensor, snd: torch.Tensor):
        return torch.cat([fst, snd], dim=1)


class BaseEncoder(nn.Module):
    def __init__(
            self,
            depth: int = 7,
            in_channels: int = 4,
            channels: int = 64,
            max_channels: int = 512,
            norm_layer: Optional[nn.Module] = nn.BatchNorm2d,
            masked_norm: bool = False,
            activation: nn.Module = nn.ELU,
            backbone_start_connect_pos: int = -1,
            backbone_channels: Optional[List[int]] = None,
            aggregation_mode: str = ''
    ):
        super(BaseEncoder, self).__init__()

        self.depth = depth
        self.backbone_start_connect_pos = backbone_start_connect_pos

        self.output_channels: List[int] = [channels, channels]

        self.block0 = ConvNormAct(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=4,
            padding=1,
            stride=2,
            norm_layer=None,
            activation=activation,
            conv_layer=nn.Conv2d,
            masked_norm=masked_norm
        )

        self.block1 = ConvNormAct(
            in_channels=channels,
            out_channels=channels,
            kernel_size=4,
            padding=1,
            stride=2,
            norm_layer=None,
            activation=activation,
            conv_layer=nn.Conv2d,
            masked_norm=masked_norm
        )

        self.block_dict = nn.ModuleDict()
        self.connect_dict = nn.ModuleDict()

        if backbone_channels is None:
            backbone_channels = []
        reversed_backbone_channels = list(reversed(backbone_channels))

        out_channels = channels
        for block_idx in range(2, depth):
            if block_idx % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)

            if 0 <= backbone_start_connect_pos <= block_idx and len(reversed_backbone_channels):
                stage_channels = reversed_backbone_channels.pop()
                connector = FeatureAggregation(in_channels, stage_channels, mode=aggregation_mode)
                in_channels = connector.out_channels
                self.connect_dict[f'connect_{block_idx}'] = connector
            self.block_dict[f'block_{block_idx}'] = ConvNormAct(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                padding=int(block_idx + 1 < depth),
                stride=2,
                norm_layer=norm_layer,
                activation=activation,
                conv_layer=nn.Conv2d,
                masked_norm=masked_norm
            )
            self.output_channels += [out_channels]

    def forward(
            self,
            comp_image: torch.Tensor,
            mask: torch.Tensor,
            backbone_feats: Optional[List[torch.Tensor]] = None
    ):
        x = torch.cat([comp_image, mask], dim=1)

        output = []

        output += [self.block0(x, mask=None)]
        output += [self.block1(output[-1], mask)]

        reversed_backbone_feats = []
        if backbone_feats is not None:
            reversed_backbone_feats = list(reversed(backbone_feats))

        for block_idx in range(2, self.depth):
            x = output[-1]
            if 0 <= self.backbone_start_connect_pos <= block_idx and len(reversed_backbone_feats):
                bb_feat = reversed_backbone_feats.pop()
                connector = self.connect_dict[f'connect_{block_idx}']
                x = connector(x, bb_feat)
            conv_block = self.block_dict[f'block_{block_idx}']
            output += [conv_block(x, mask)]
        return output


class BaseDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels: List[int],
            depth: int = 7,
            norm_layer: Optional[nn.Module] = nn.BatchNorm2d,
            masked_norm: bool = False,
            activation: nn.Module = nn.ELU,
            image_fusion: bool = True
    ):
        super(BaseDecoder, self).__init__()
        self.image_fusion = image_fusion
        self.deconv_list = nn.ModuleList()
        self.depth = depth

        encoder_channels = copy(encoder_channels)

        in_channels = encoder_channels.pop()

        for block_idx in range(depth):
            if len(encoder_channels):
                out_channels = encoder_channels.pop()
            else:
                out_channels = in_channels // 2
            self.deconv_list.append(
                ConvNormAct(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=int(block_idx > 0),
                    norm_layer=norm_layer,
                    activation=activation,
                    conv_layer=nn.ConvTranspose2d,
                    masked_norm=masked_norm
                )
            )
            in_channels = out_channels
        self.to_rgb = nn.Conv2d(in_channels, 3, kernel_size=1)

        if image_fusion:
            self.conv_attention = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(
            self,
            encoder_output: List[torch.Tensor],
            comp_image: torch.Tensor,
            mask: torch.Tensor
    ):
        """
        it's guarantee that len(encoder_output) == len(self.deconv_list) by __init__
        """
        x = encoder_output.pop()
        for block in self.deconv_list[:-1]:
            skip_connection = encoder_output.pop()
            x = block(x, mask) + skip_connection
        x = self.deconv_list[-1](x, mask)
        harm_image = self.to_rgb(x)
        if self.image_fusion:
            attention_map = (3 * self.conv_attention(x)).sigmoid()
            harm_image = attention_map * comp_image + (1.0 - attention_map) * harm_image
        return harm_image
