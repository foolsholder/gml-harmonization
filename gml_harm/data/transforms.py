from albumentations import Normalize

import torch

import numpy as np

from typing import Tuple


class ToOriginalScale:
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 max_pixel_value=255.):
        self.mean = torch.Tensor(mean) * max_pixel_value
        self.std = torch.Tensor(std) * max_pixel_value

        self.mean = self.mean.reshape(1, 3, 1, 1)
        self.std = self.std.reshape(1, 3, 1, 1)

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        tensor.shape == [batch_size, 3, H, W]
        """
        device = tensor.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        tensor = tensor * std + mean
        return tensor


class LookUpTable:
    def __init__(self):
        pass

    def __call__(self,
                 content: np.array,
                 reference: np.array) -> Tuple[np.array, np.array]:
        return content, reference
