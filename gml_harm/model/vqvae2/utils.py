import torch

from collections import OrderedDict
from typing import Any, OrderedDict as ORDType

from .vqvae2 import HVQVAE


def create_hvqvae(model_cfg: ORDType[str, Any]) -> torch.nn.Module:
    model = HVQVAE(**model_cfg)

    result_dict: ORDType[str, Any] = OrderedDict()

    result_dict['content_encoder'] = getattr(model, 'content_encoder')
    result_dict['reference_encoder'] = getattr(model, 'reference_encoder')
    result_dict['decoder'] = getattr(model, 'decoder')

    result_dict['model'] = model
    return result_dict
