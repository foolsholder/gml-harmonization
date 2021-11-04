import torch

from torch import nn


class PSNR(nn.Module):
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        batch_size = outputs.size(0)
        outputs = outputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        mse = torch.mean((outputs - targets) ** 2, dim=1)

        psnr = 10 * torch.log10(255 ** 2 / mse)

        return psnr
