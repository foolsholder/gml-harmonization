import torch

from torch import nn
from typing import Tuple


class ForegroundNormalizedMSE(nn.Module):
    def __init__(self, min_area: float = 100.):
        super(ForegroundNormalizedMSE, self).__init__()
        self.min_area = min_area

    def forward(self, outputs: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        gt_images, masks = targets
        batch_size = masks.size(0)
        delimeter = torch.maximum(torch.sum(masks.view(batch_size, -1), dim=1), self.min_area)

        mask_l2 = masks * (outputs - gt_images) ** 2

        fnmse = mask_l2.view(batch_size, -1).sum(dim=1) / delimeter
        
        return fnmse
    