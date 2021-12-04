from copy import copy
from collections import OrderedDict
from torch import nn
from typing import Dict, Any, OrderedDict as ORDType

from gml_harm.model.vqvae2 import create_hvqvae
from gml_harm.model.idih import create_dih


def create_model(model_cfg: ORDType[str, Any]) -> ORDType[str, nn.Module]:
    possible_models = {
        'HVQVAE': create_hvqvae,
        'DIH': create_dih
    }
    model_cfg = copy(model_cfg)
    model_type_name = model_cfg.pop('type')
    create_nn = possible_models[model_type_name]
    model: OrderedDict[str, nn.Module] = create_nn(model_cfg)
    return model
