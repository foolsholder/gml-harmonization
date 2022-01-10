from collections import OrderedDict
from typing import Any, OrderedDict as ORDType, Union

import torch.nn

from .swin import SwinTransformer


def create_swin_transformer(swin_cfg: ORDType[str, Any]) -> torch.nn.Module:
    pretrained_checkpoint_path = ''
    if 'pretrained' in swin_cfg:
        pretrained_checkpoint_path = swin_cfg.pop('pretrained')
    swin = SwinTransformer(**swin_cfg)
    if pretrained_checkpoint_path:
        swin.load_state_dict(torch.load(pretrained_checkpoint_path), strict=False)
    return swin
