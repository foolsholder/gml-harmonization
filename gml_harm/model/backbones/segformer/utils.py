import torch
from torch import nn
from typing import OrderedDict as ORDType, Any
from .segformer import SegFormer


def create_segformer(segformer_cfg: ORDType[str, Any]) -> nn.Module:
    pretrained_checkpoint_path = ''
    if 'pretrained' in segformer_cfg:
        pretrained_checkpoint_path = segformer_cfg.pop('pretrained')
    segformer = SegFormer(**segformer_cfg)
    if pretrained_checkpoint_path:
        segformer.load_state_dict(torch.load(pretrained_checkpoint_path), strict=False)
    return segformer