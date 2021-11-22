import glob

import cv2
import torch

import numpy as np

from albumentations import Compose
from copy import copy
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict, Union, List, Callable

from .base_dataset import ABCDataset
from.transforms import LookUpTable


class SSHTrainDataset(ABCDataset):
    def __init__(self,
                 dataset_path: str,
                 split: str = 'train',
                 augmentations: Compose = None,
                 crop: Compose = None,
                 to_tensor_transform: Compose = None,
                 LUT: LookUpTable = None):
        super(SSHTrainDataset, self).__init__()

        self.dataset_path = Path(dataset_path)

        assert split == 'train'
        assert crop is not None
        assert to_tensor_transform is not None
        assert LUT is not None

        self.augmentations = augmentations
        self.crop = crop
        self.to_tensor_transform = to_tensor_transform

        self.LUT = LUT
        images = self.dataset_path / 'real_images'
        self.dataset_samples: List[Any] = list(images.glob('*.jpg'))

    def __len__(self) -> int:
        return len(self.dataset_samples)

    def get_sample(self, idx) -> Dict[str, Union[np.array, str]]:
        image_path = self.dataset_samples[idx]
        image: np.array = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        out: Dict[str, Union[np.array, str]] = {
            'image': image,
        }
        return out

    def check_sample_types(self, sample: Dict[str, Union[np.array, str]]) -> None:
        assert sample['image'].dtype == 'uint8'
        if 'target' in sample:
            assert sample['target'].dtype == 'uint8'

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, np.array, str]]:
        sample = self.get_sample(idx)
        self.check_sample_types(sample)
        sample = self.augment_sample(sample)

        image = sample['image']
        crop_content = self.crop(image=image)['image']
        crop_reference = self.crop(image=image)['image']

        content_alpha, reference_alpha = self.LUT(crop_content, crop_reference)
        content_beta, reference_beta = self.LUT(crop_content, crop_reference)

        out = {
            'image': content_alpha,
            'reference_alpha': reference_alpha,
            'content_beta': content_beta,
            'reference_beta': reference_beta
        }

        out = self.to_tensor_transform(**out)
        content_alpha_tensor = out.pop('image')
        out['content_alpha'] = content_alpha_tensor

        return out

    def augment_sample(self, sample) -> Dict[str, Union[np.array, str]]:
        if self.augmentations is not None:
            sample.update(self.augmentations(image=sample['image']))
        return sample


class SSHTestDataset(ABCDataset):
    def __init__(self,
                 dataset_path: str,
                 split: str = 'test',
                 to_tensor_transform: Compose = None):
        super(SSHTestDataset, self).__init__()

        self.dataset_path = Path(dataset_path)

        assert split == 'test'
        assert to_tensor_transform is not None

        self.to_tensor_transform = to_tensor_transform

        self.content_images = self.dataset_path / 'content_images'
        self.reference_images = self.dataset_path / 'reference_image'
        self.gt_images = self.dataset_path / 'harmonized_images'
        self.masks = self.dataset_path / 'ssh_masks'
        self.dataset_samples: List[Any] = list(self.content_images.glob('*.jpg'))

    def __len__(self) -> int:
        return len(self.dataset_samples)

    def _imread(self, img_path, bgr2rgb: bool = False):
        img = cv2.imread(img_path)
        if bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_sample(self, idx) -> Dict[str, Union[np.array, str]]:
        image_name = self.dataset_samples[idx].name

        content_image = self._imread(self.content_images / image_name, bgr2rgb=True)
        reference_image = self._imread(self.reference_images / image_name, bgr2rgb=True)
        gt_image = self._imread(self.gt_images / image_name, bgr2rgb=True)
        mask = self._imread(self.masks / image_name, bgr2rgb=False)

        out: Dict[str, Union[np.array, str]] = {
            'image': content_image,
            'reference_image': reference_image,
            'target': gt_image,
            'mask': mask
        }
        return out

    def check_sample_types(self, sample: Dict[str, Union[np.array, str]]) -> None:
        assert sample['image'].dtype == 'uint8'
        if 'target' in sample:
            assert sample['target'].dtype == 'uint8'

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, np.array, str]]:
        sample = self.get_sample(idx)
        self.check_sample_types(sample)

        sample = self.to_tensor_transform(**sample)

        out = {
            'content_images': sample['image'],
            'reference_images': sample['reference_image'],
            'masks': sample['mask'],
            'targets': sample['target']
        }

        return out
