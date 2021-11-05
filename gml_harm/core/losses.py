import torch

from torch import nn
from typing import Tuple

from .functional import mse, psnr, fn_mse


class MSE(nn.Module):
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return mse(outputs, targets)


class PSNR(nn.Module):
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return psnr(outputs, targets)


class ForegroundNormalizedMSE(nn.Module):
    def __init__(self, min_area: float = 100.):
        super(ForegroundNormalizedMSE, self).__init__()
        self.min_area = min_area

    def forward(self, outputs: torch.Tensor, targets_and_masks: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return fn_mse(outputs, targets_and_masks)
