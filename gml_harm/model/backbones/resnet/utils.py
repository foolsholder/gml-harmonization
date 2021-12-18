from collections import OrderedDict
from typing import Any, OrderedDict as ORDType, Union

import torch.nn

from .resnet import ResNetV1c, ResNet


def create_resnet(resnet_cfg: ORDType[str, Any]) -> torch.nn.Module:
    pretrained_checkpoint_path = ''
    if 'pretrained' in resnet_cfg:
        pretrained_checkpoint_path = resnet_cfg.pop('pretrained')
    resnet = ResNetV1c(**resnet_cfg)
    if pretrained_checkpoint_path:
        resnet.load_state_dict(torch.load(pretrained_checkpoint_path))
    return resnet
