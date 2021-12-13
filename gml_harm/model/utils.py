from copy import deepcopy
from collections import OrderedDict
from torch import nn
from typing import Dict, Any, OrderedDict as ORDType

from gml_harm.model.vqvae2 import create_hvqvae
from gml_harm.model.idih import create_dih

from gml_harm.core.layers.rain import RAIN


class _ARGS_MAP:
    POSSIBLE_NORMALIZATION = {
        "RAIN": RAIN,
        "BatchNorm2d": nn.BatchNorm2d,
        "InstanceNorm2d": nn.InstanceNorm2d
    }


def _dfs_correct_cfg(submodel_cfg: ORDType[str, Any]) -> None:
    order_keys = []
    for k, v in submodel_cfg.items():
        if isinstance(v, dict):
            _dfs_correct_cfg(v)
        elif isinstance(v, str):
            if 'norm_layer' in k:
                order_keys += [k, 'norm_layer']

    for k, type in order_keys:
        if type == 'norm_layer':
            v = submodel_cfg[k]
            submodel_cfg[k] = _ARGS_MAP.POSSIBLE_NORMALIZATION[v]


def create_model(model_cfg: ORDType[str, Any]) -> ORDType[str, nn.Module]:
    possible_models = {
        'HVQVAE': create_hvqvae,
        'DIH': create_dih
    }
    model_cfg = deepcopy(model_cfg)

    model_type_name = model_cfg.pop('type')
    create_nn = possible_models[model_type_name]
    model: OrderedDict[str, nn.Module] = create_nn(model_cfg)
    return model
