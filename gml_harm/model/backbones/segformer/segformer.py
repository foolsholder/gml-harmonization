import torch
from torch import nn
from typing import OrderedDict as ORDType, Any, List

from .segformer_entities import (
    mit_b0,
    mit_b1,
    mit_b2,
    mit_b3,
    mit_b4,
    mit_b5
)
from .segformer_head import SegFormerHead


def create_segbb(bb_cfg: ORDType[str, Any]) -> nn.Module:
    possible_bb = {
        'mit_b0': mit_b0,
        'mit_b1': mit_b1,
        'mit_b2': mit_b2,
        'mit_b3': mit_b3,
        'mit_b4': mit_b4,
        'mit_b5': mit_b5,
    }
    typename = bb_cfg.pop('type')
    typeclass = possible_bb[typename]
    model = typeclass(**bb_cfg)
    return model


def create_seghead(head_cfg: ORDType[str, Any]) -> nn.Module:
    return SegFormerHead(**head_cfg)


class SegFormer(nn.Module):
    def __init__(self, bb_cfg: ORDType[str, Any], head_cfg: ORDType[str, Any]):
        super(SegFormer, self).__init__()
        self.backbone = create_segbb(bb_cfg)
        self.decode_head = create_seghead(head_cfg)

    def forward(self, x, mask=None) -> List[torch.Tensor]:
        return self.decode_head(self.backbone(x))
