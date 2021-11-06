import torch

import numpy as np

from typing import Tuple


def mse(outputs: torch.Tensor, targets: torch.Tensor, reduce=True) -> torch.Tensor:
    """
    :param outputs: [batch_size, C, H, W]
    :param targets: [batch_size, C, H, W]
    :return: [batch_size,]
    """
    batch_size = outputs.size(0)
    outputs = outputs.view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    mse_res = torch.mean((outputs - targets) ** 2, dim=1)
    if reduce:
        return torch.mean(mse_res)
    return mse_res


def psnr(outputs: torch.Tensor,
         targets: torch.Tensor,
         max_pixel_value=255.0) -> torch.Tensor:
    """
    :param outputs: [batch_size, C, H, W]
    :param targets: [batch_size, C, H, W]
    :return: [batch_size,]
    """
    mse_res = mse(outputs, targets, reduce=False)
    psnr_res = 10 * torch.log10(max_pixel_value ** 2 / mse_res)
    return torch.mean(psnr_res)


def fmse(outputs: torch.Tensor,
           targets_and_masks: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    batch_size = outputs.size(0)
    (targets, masks) = targets_and_masks
    mask_l2_square = masks * ((outputs - targets) ** 2)
    mask_l2_square = mask_l2_square.view(batch_size, -1)
    mask_l2_square = torch.sum(mask_l2_square, dim=1)

    masks = masks.view(batch_size, -1)
    denominator = torch.sum(masks, dim=1)

    zero32 = np.float32(0)

    denominator_mask = denominator > zero32
    inv_denominator_mask = torch.logical_not(denominator_mask)
    add_values = inv_denominator_mask.float()
    denominator = denominator + add_values
    fmse = torch.divide(mask_l2_square, denominator)
    fmse = torch.where(denominator_mask, fmse, mask_l2_square)
    return torch.mean(fmse)


def fn_mse(outputs: torch.Tensor,
           targets_and_masks: Tuple[torch.Tensor, torch.Tensor],
           min_area=100.) -> torch.Tensor:
    batch_size = outputs.size(0)
    (targets, masks) = targets_and_masks
    mask_l2_square = masks * ((outputs - targets) ** 2)
    mask_l2_square = mask_l2_square.view(batch_size, -1)
    mask_l2_square = torch.sum(mask_l2_square, dim=1)

    masks = masks.view(batch_size, -1)
    denominator = torch.clamp_min(torch.sum(masks, dim=1), min_area)
    fn_mse = mask_l2_square / denominator
    return torch.mean(fn_mse)