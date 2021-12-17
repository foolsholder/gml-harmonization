from collections import OrderedDict
from typing import Any, OrderedDict as ORDType, Union

import torch.nn

from .cswin import CSWin


def create_cswin_transformer(cswin_cfg: ORDType[str, Any]) -> torch.nn.Module:
    pretrained_checkpoint_path = ''
    if 'pretrained' in cswin_cfg:
        pretrained_checkpoint_path = cswin_cfg.pop('pretrained')
    cswin = CSWin(**cswin_cfg)
    if pretrained_checkpoint_path:
        cswin.load_state_dict(torch.load(pretrained_checkpoint_path))
    return cswin
