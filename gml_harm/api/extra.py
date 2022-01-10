import os.path

import torch
import torch.distributed as dist

from catalyst import dl
from collections import OrderedDict
from typing import Dict, Any, List, Optional

from .train import train_model


def train_swa(model, cfg: Dict[str, Any]):
    assert 'resume' in cfg
    resume: str = cfg.pop('resume')


    """
    TODO:
    here transform config to train SWA
    """

    """
    the same as in train
    """

