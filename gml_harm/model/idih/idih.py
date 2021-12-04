import torch

from collections import OrderedDict
from torch import nn
from typing import List, OrderedDict as ORDType, Any

from .idih_entities import BaseEncoder, BaseDecoder
from ..backbones.swin2.swin2 import SwinV2
from ..backbones.utils import create_backbone


class DIH(nn.Module):
    def __init__(
            self
    ):
        super(DIH, self).__init__()
        self.encoder = BaseEncoder()
        encoder_channels: List[int] = self.encoder.output_channels
        self.decoder = BaseDecoder(encoder_channels=encoder_channels)

    def forward(self, comp_image: torch.Tensor, mask: torch.Tensor):
        enc_output = self.encoder(comp_image, mask, backbone_feats=None)
        harm_image = self.decoder(enc_output, comp_image, mask)
        return harm_image


class BackbonedDIH(nn.Module):
    def __init__(
            self,
            backbone_cfg: OrderedDict[str, Any]
    ):
        super(BackbonedDIH, self).__init__()
        self.backbone = create_backbone(backbone_cfg)
        self.base = DIH()

    def forward(self, comp_image: torch.Tensor, mask: torch.Tensor):
        backbone_feats = self.backbone(comp_image, mask)
        harm_image = self.base(comp_image, mask, backbone_feats)
        return harm_image
