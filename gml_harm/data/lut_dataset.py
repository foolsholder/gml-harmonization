import torch
import cv2

import albumentations as A
import numpy as np

from copy import deepcopy, copy
from torch.utils.data import Dataset
from typing import Union, Dict, List, Tuple
from pathlib import Path
from .lut.lookuptable import LookUpTable3D

from .base_dataset import ABCDataset


class LutDataset(ABCDataset):
    def __init__(
            self,
            dataset_path: Union[Path, str],
            luts_dir: Union[Path, str],
            split: str = 'train',
            geometric_augmentations: A.Compose = None,
            color_augmentations: A.Compose = None,
            crop: A.Compose = None,
            to_tensor_transforms: A.Compose = None,
            keep_without_mask: float = 0.05
    ):
        super(LutDataset, self).__init__()
        self.dataset_path = Path(dataset_path)
        self.keep_without_mask = keep_without_mask

        self.luts: List[LookUpTable3D] = []
        for lut_file in Path(luts_dir).iterdir():
            self.luts += [LookUpTable3D(lut_file)]

        self.geometric_augmentations = geometric_augmentations
        self.color_augmentations = color_augmentations
        self.crop = crop
        self.to_tensor_transforms = to_tensor_transforms

        self.mask_dir = self.dataset_path / 'masks'
        self.image_dir = self.dataset_path / 'real_images'
        self.dataset_samples: List[Path] = list(self.mask_dir.glob('*.png'))

    def __len__(self) -> int:
        return len(self.dataset_samples)

    def get_sample(self, idx: int) -> Dict[str, Union[np.array, str]]:
        mask_path = str(self.dataset_samples[idx])
        mask_name = mask_path.split('/')[-1][:-4]
        image_name = mask_name.split('_')[0]
        image_path = str(self.image_dir / (image_name + '.jpg'))

        image: np.array = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask: np.array = cv2.imread(mask_path)
        mask = mask[:, :, 0].astype(np.float32)
        mask /= mask.max()

        out: Dict[str, Union[np.array, str]] = {
            'image': image,
            'mask': mask,
        }
        return out

    def apply_lut(
            self,
            image: np.array,
            ret_tensor: bool = False
    ) -> Union[torch.Tensor, np.array]:
        """
        cpu LUT
        """
        image = image.astype(np.float32) / 255.
        lut = np.random.choice(self.luts)
        tensor = torch.FloatTensor(image).permute(2, 0, 1)[None]
        tensor = lut(tensor)
        if ret_tensor:
            return tensor
        image = tensor[0].permute(1, 2, 0).data.numpy()
        return image

    # noinspection PyTypeChecker
    def __getitem__(self, idx: int) -> Dict[str, Union[np.array, torch.Tensor, str]]:
        sample: Dict[str, Union[np.array, torch.Tensor, str]] = self.get_sample(idx)
        self.check_sample_types(sample)
        sample = self.augment_sample(sample, self.geometric_augmentations)

        background = self.color_augmentations(**sample)['image']
        foreground = self.color_augmentations(**sample)['image']

        background = self.apply_lut(background, ret_tensor=False)
        foreground = self.apply_lut(foreground, ret_tensor=False)

        composite = foreground * sample['mask'][:, :, None] + \
                    background * (1. - sample['mask'][:, :, None])
        out = {
            'image': composite * 255,
            'mask': sample['mask'],
            'target': background * 255
        }

        out = self.to_tensor_transforms(**out)
        out = {
            'images': out['image'],
            'masks': out['mask'][None],
            'targets': out['target']
        }
        return out

    def check_sample_types(self, sample: Dict[str, Union[np.array, str]]) -> None:
        assert sample['image'].dtype == 'uint8'
        if 'target' in sample:
            assert sample['target'].dtype == 'uint8'

    def __len__(self) -> int:
        return len(self.dataset_samples)

    def augment_sample(self,
                       sample: Dict[str, Union[np.array, str]],
                       augmentations: A.Compose) -> Dict[str, Union[np.array, str]]:
        if augmentations is None:
            augmentations = A.Compose([], p=1.0)

        sample = copy(sample)

        additional_targets: Dict[str, np.array] = {target_name: sample[target_name]
                                                   for target_name in augmentations.additional_targets.keys()
                                                   if target_name in sample}

        valid_augmentation: bool = False
        cropped_output: Dict[str, np.array] = {}
        while not valid_augmentation:
            cropped_output = self.crop(image=sample['image'],
                                       mask=sample['mask'],
                                       **additional_targets)
            valid_augmentation = self.check_augmented_sample(cropped_output)
        aug_output = augmentations(**cropped_output)

        for target_name, transformed_target in aug_output.items():
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, aug_output: Dict[str, np.array]) -> bool:
        if self.keep_without_mask > 0. and np.random.rand() < self.keep_without_mask:
            return True
        return aug_output['mask'].sum() > 1.0
