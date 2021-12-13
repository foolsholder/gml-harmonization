import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import MSELoss as MSECriterion
from typing import Tuple


class PSNRCriterion(MSECriterion):
    def __init__(self, max_pixel_value=255., **kwargs):
        super(PSNRCriterion, self).__init__(**kwargs)
        self.max_pixel_value = max_pixel_value

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        batch_size = input.size(0)
        mse = F.mse_loss(input, target, reduction='none').view(batch_size, -1).mean(dim=1)
        psnr = 10 * torch.log10(target.view(batch_size, -1).max(dim=1).values ** 2 / mse)
        return psnr.mean()


class fMSECriterion(MSECriterion):
    def forward(self, input: Tensor, target: Tuple[Tensor, Tensor]) -> Tensor:
        target, mask = target
        batch_size = input.size(0)
        channels = input.size(1)
        sse = F.mse_loss(input, target, reduction='none').view(batch_size, -1).sum(dim=1)
        mask = channels * mask.view(batch_size, -1).sum(dim=1)
        fmse = sse / mask
        return fmse.mean()


class FNMSECriterion(MSECriterion):
    def __init__(self, min_area: float = 100.):
        super(FNMSECriterion, self).__init__()
        self.min_area = min_area

    def forward(self, input: Tensor, target: Tuple[Tensor, Tensor]) -> Tensor:
        target, mask = target
        batch_size = input.size(0)
        channels = input.size(1)
        sse = F.mse_loss(input, target, reduction='none').view(batch_size, -1).sum(dim=1)
        mask = mask.view(batch_size, -1).sum(dim=1)
        mask = channels * torch.clamp_min(mask, self.min_area)
        fmse = sse / mask
        return fmse.mean()
