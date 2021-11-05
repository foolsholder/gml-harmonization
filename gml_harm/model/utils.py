from copy import copy
from torch import nn

from .unet import UNet


def create_model(model_cfg) -> nn.Module:
    possible_models = {
        'UNet': UNet
    }
    model_cfg = copy(model_cfg)
    model_type_name = model_cfg.pop('type')
    model_type = possible_models[model_type_name]
    model = model_type(model_cfg)
    return model