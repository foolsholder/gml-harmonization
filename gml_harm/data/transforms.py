from albumentations import Normalize

import torch

import numpy as np

from typing import Tuple, List, Union
from pathlib import Path

from albumentations import DualTransform
from albumentations.imgaug.transforms import DualIAATransform
from albumentations import to_tuple
import imgaug.augmenters as iaa

from .lut.lookuptable import LookUpTable3D


class IAAAffine2(DualIAATransform):
    """Place a regular grid of points on the input and randomly move the neighbourhood of these point around
    via affine transformations.
    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask
    """

    def __init__(
        self,
        scale=(0.7, 1.3),
        translate_percent=None,
        translate_px=None,
        rotate=0.0,
        shear=(-0.1, 0.1),
        order=1,
        cval=0,
        mode="reflect",
        always_apply=False,
        p=0.5,
    ):
        super(IAAAffine2, self).__init__(always_apply, p)
        self.scale = dict(x=scale, y=scale)
        self.translate_percent = to_tuple(translate_percent, 0)
        self.translate_px = to_tuple(translate_px, 0)
        self.rotate = to_tuple(rotate)
        self.shear = dict(x=shear, y=shear)
        self.order = order
        self.cval = cval
        self.mode = mode

    @property
    def processor(self):
        return iaa.Affine(
            self.scale,
            self.translate_percent,
            self.translate_px,
            self.rotate,
            self.shear,
            self.order,
            self.cval,
            self.mode,
        )

    def get_transform_init_args_names(self):
        return ("scale", "translate_percent", "translate_px", "rotate", "shear", "order", "cval", "mode")


class IAAPerspective2(DualIAATransform):
    """Perform a random four point perspective transform of the input.
    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}
    Args:
        scale ((float, float): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners. Default: (0.05, 0.1).
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask
    """

    def __init__(self, scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5,
                 order=1, cval=0, mode="replicate"):
        super(IAAPerspective2, self).__init__(always_apply, p)
        self.scale = to_tuple(scale, 1.0)
        self.keep_size = keep_size
        self.cval = cval
        self.mode = mode

    @property
    def processor(self):
        return iaa.PerspectiveTransform(self.scale, keep_size=self.keep_size, mode=self.mode, cval=self.cval)

    def get_transform_init_args_names(self):
        return ("scale", "keep_size")


class ToOriginalScale(torch.nn.Module):
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 max_pixel_value=255.):
        super(ToOriginalScale, self).__init__()
        mean = torch.Tensor(mean) * max_pixel_value
        std = torch.Tensor(std) * max_pixel_value

        self.register_buffer("mean", mean.reshape(1, 3, 1, 1))
        self.register_buffer("std", std.reshape(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        tensor.shape == [batch_size, 3, H, W]
        """
        self.to(tensor)
        tensor = tensor * self.std + self.mean
        return tensor


class LookUpTableAUG(DualTransform):
    def __init__(
        self,
        luts_dir: str='../np_store/',
        always_apply: bool=False,
        p: float=0.5,
    ):
        super(LookUpTableAUG, self).__init__(always_apply, p)
        self.luts: List[LookUpTableAUG] = []
        for fname in Path(luts_dir).iterdir():
            self.luts += [LookUpTable3D(fname)]

    def apply_to_mask(self, img, **params):
        return img

    def apply(self, img, lut_idx=-1, **params):
        """
        input img_max_value = 255.
        output img_max_value = 255.
        """
        if lut_idx == -1:
            return img
        lut = self.luts[lut_idx]
        img = img.astype(np.float32) / 255.
        tensor = torch.FloatTensor(img).permute(2, 0, 1)[None]
        tensor = lut(tensor)
        img = tensor[0].permute(1, 2, 0).data.numpy()
        img = np.around(np.clip(img, 0, 1) * 255, decimals=0)
        img = img.astype(np.uint8)
        return img

    def get_params(self):
        return {
            "lut_idx": np.random.randint(len(self.luts))
        }