import torch

from collections import OrderedDict
from torch import nn
from typing import List, Any, Optional, OrderedDict as ORDType

from .idih_entities import BaseEncoder, BaseDecoder
from ..backbones.swin2.swin2 import SwinV2
from ..backbones.utils import create_backbone


class DIH(nn.Module):
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
            aggregation_mode: str = '',
            image_fusion: bool = True
    ):
        super(DIH, self).__init__()
        self.encoder = BaseEncoder(
            depth=depth,
            in_channels=in_channels,
            channels=channels,
            max_channels=max_channels,
            norm_layer=norm_layer,
            masked_norm=masked_norm,
            activation=activation,
            backbone_start_connect_pos=backbone_start_connect_pos,
            backbone_channels=backbone_channels,
            aggregation_mode=aggregation_mode
        )
        encoder_channels: List[int] = self.encoder.output_channels
        self.decoder = BaseDecoder(
            encoder_channels=encoder_channels,
            depth=depth,
            norm_layer=norm_layer,
            masked_norm=masked_norm,
            activation=activation,
            image_fusion=image_fusion
        )

    def forward(self, comp_image: torch.Tensor, mask: torch.Tensor):
        enc_output = self.encoder(comp_image, mask, backbone_feats=None)
        harm_image = self.decoder(enc_output, comp_image, mask)
        return harm_image


class BackbonedDIH(nn.Module):
    def __init__(
            self,
            backbone_cfg: ORDType[str, Any],
            **kwargs
    ):
        super(BackbonedDIH, self).__init__()
        self.backbone = create_backbone(backbone_cfg)
        self.base = DIH(**kwargs)

    def forward(self, comp_image: torch.Tensor, mask: torch.Tensor):
        backbone_feats = self.backbone(comp_image, mask)
        harm_image = self.base(comp_image, mask, backbone_feats)
        return harm_image
