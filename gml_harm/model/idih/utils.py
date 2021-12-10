import torch

from collections import OrderedDict
from typing import Any, OrderedDict as ORDType

from .idih import DIH, BackbonedDIH
from gml_harm.core.weights_initializer import XavierGluon


def create_dih(model_cfg: ORDType[str, Any]) -> torch.nn.Module:
    use_bb: bool = False
    bb_cfg: ORDType[str, Any] = OrderedDict()
    if 'backbone' in model_cfg:
        bb_cfg = model_cfg.pop('backbone')
        use_bb = bb_cfg.pop('use') if len(bb_cfg) else False
    base_cfg: ORDType[str, Any] = model_cfg.pop('base')

    result: ORDType[str, torch.nn.Module] = OrderedDict()
    if not use_bb:
        model = DIH(**base_cfg)
        model.apply(XavierGluon(rnd_type='gaussian', magnitude=2.0))
        # result['base'] = model
    else:
        model = BackbonedDIH(bb_cfg, **base_cfg)
        result['base'] = getattr(model, 'base')
        result['backbone'] = getattr(model, 'backbone')

    result['model'] = model
    return result