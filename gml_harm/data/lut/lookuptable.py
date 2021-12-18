import torch

import numpy as np

from torch import nn
from typing import Union
from pathlib import Path

from .lut_entities import TrilinearInterpolation


class LookUpTable3D(nn.Module):
    def __init__(self, lut_path: Union[Path, str]):
        super(LookUpTable3D, self).__init__()
        buffer = np.load(str(lut_path))
        """
        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k
                    x = lines[n].split()
                    buffer[0, i, j, k] = float(x[0])
                    buffer[1, i, j, k] = float(x[1])
                    buffer[2, i, j, k] = float(x[2])
        """
        self.register_buffer('LUT', torch.from_numpy(buffer))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.LUT = self.LUT.to(x)
        _, output = self.TrilinearInterpolation(self.LUT, x)
        return output
