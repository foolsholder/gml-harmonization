from collections import OrderedDict
from typing import Any, OrderedDict as ORDType, Dict, Callable

import torch.nn

from .swin.utils import create_swin_transformer
from .resnet.utils import create_resnet
from .cswin.utils import create_cswin_transformer


def create_backbone(backbone_cfg: ORDType[str, Any]) -> torch.nn.Module:
    possible_bb: Dict[str, Callable[[ORDType[str, Any]], torch.nn.Module]] = {
        "Swin": create_swin_transformer,
        "CSwin": create_cswin_transformer,
        "Resnet": create_resnet
    }
    type_name = backbone_cfg.pop('type')
    create_bb = possible_bb[type_name]
    return create_bb(backbone_cfg)