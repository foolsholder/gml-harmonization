import torch

from typing import Tuple


def mse(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    :param outputs: [batch_size, C, H, W]
    :param targets: [batch_size, C, H, W]
    :return: [batch_size,]
    """
    batch_size = outputs.size(0)
    outputs = outputs.view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    mse_res = torch.mean((outputs - targets) ** 2, dim=1)
    return mse_res


def psnr(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    :param outputs: [batch_size, C, H, W]
    :param targets: [batch_size, C, H, W]
    :return: [batch_size,]
    """
    mse_res = mse(outputs, targets)
    psnr_res = 10 * torch.log10(255 ** 2 / mse)
    return psnr_res


def fn_mse(outputs: torch.Tensor,
           targets_and_masks: Tuple[torch.Tensor, torch.Tensor],
           min_area=100.) -> torch.Tensor:
    batch_size = outputs.size(0)
    (targets, masks) = targets_and_masks
    mask_l2_square = masks * ((outputs - targets) ** 2)
    mask_l2_square = mask_l2_square.view(batch_size, -1)
    mask_l2_square = torch.sum(mask_l2_square, dim=1)

    masks = masks.view(batch_size, -1)
    denominator = torch.maximum(torch.sum(masks, dim=1), min_area)

    fn_mse = mask_l2_square / denominator
    return fn_mse