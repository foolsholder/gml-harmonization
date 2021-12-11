import torch

from collections import OrderedDict
from typing import Any, OrderedDict as ORDType

from .idih import DIH, BackbonedDIH
from gml_harm.core.weights_initializer import XavierGluon


def create_dih(model_cfg: ORDType[str, Any]) -> torch.nn.Module:
    use_bb: bool = False
    bb_cfg: ORDType[str, Any] = OrderedDict()
    mask_fusion_cfg: str = ''
    if 'backbone' in model_cfg:
        bb_cfg = model_cfg.pop('backbone')
        mask_fusion_cfg = model_cfg.pop('mask_fusion_bb')
        use_bb = True
    base_cfg: ORDType[str, Any] = model_cfg.pop('base')

    result: ORDType[str, torch.nn.Module] = OrderedDict()
    if not use_bb:
        model = DIH(**base_cfg)
        # model.apply(XavierGluon(rnd_type='gaussian', magnitude=2.0))
        # result['base'] = model
    else:
        model = BackbonedDIH(bb_cfg, mask_fusion_cfg, **base_cfg)
        result['base'] = getattr(model, 'base')
        result['backbone'] = getattr(model, 'backbone')
        result['mask_fusion_bb'] = getattr(model, 'mask_fusion_bb')

    result['model'] = model
    return result