from copy import copy
from torch import nn
from typing import Dict, Any

from gml_harm.model.vqvae2.vqvae2 import HVQVAE

def create_model(model_cfg: Dict[str, Any]) -> nn.Module:
    possible_models = {
        'HVQVAE': HVQVAE
    }
    model_cfg = copy(model_cfg)
    model_type_name = model_cfg.pop('type')
    model_format = model_cfg.pop('format')
    model_type = possible_models[model_type_name]
    if model_format == 'single_model':
        model = model_type(**model_cfg)
    else:
        model_dict: Dict[str, str] = model_cfg.pop('model+dict')
        model = model_type(**model_cfg)
        result_dict = {}
        for k, v in model_dict.items():
            result_dict[k] = getattr(model, v)
        result_dict['model'] = model
        model = result_dict
    return model
