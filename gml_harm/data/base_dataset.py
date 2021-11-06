import cv2
import torch

import numpy as np

from albumentations import Compose
from copy import copy
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict, Union, List


class BaseDataset(Dataset):
    def __init__(self,
                 augmentations: Compose = None,
                 to_tensor_transforms: Compose = None,
                 keep_without_mask: float = 0.) -> None:
        """
        default Compose.additional_targets = {
            'target': 'image'
        }
        """
        self.augmentations = augmentations

        if to_tensor_transforms is None:
            to_tensor_transforms = lambda x: x

        self.to_tensor_transforms = to_tensor_transforms
        self.dataset_samples: List[Any] = []

        self.keep_without_mask = keep_without_mask

    # noinspection PyTypeChecker
    def __getitem__(self, idx: int) -> Dict[str, Union[np.array, torch.Tensor, str]]:
        sample: Dict[str, Union[np.array, torch.Tensor, str]] = self.get_sample(idx)
        self.check_sample_types(sample)
        sample = self.augment_sample(sample)

        sample = self.to_tensor_transforms(**sample)
        sample['mask'] = sample['mask'][None, :, :]

        sample_info = ['composite_path', 'gt_path', 'mask_path', 'image_idx']
        sample_info_dict = {info: sample[info] for info in sample_info}
        out: Dict[str, Union[np.array, torch.Tensor, str]] = {
            'images': sample['image'],
            'targets': sample['target'],
            'masks': sample['mask']
        }
        out.update(sample_info_dict)
        return out

    def get_sample(self, idx: int) -> Dict[str, Union[np.array, str]]:
        raise "You should use HDataset"

    def check_sample_types(self, sample: Dict[str, Union[np.array, str]]) -> None:
        assert sample['image'].dtype == 'uint8'
        if 'target' in sample:
            assert sample['target'].dtype == 'uint8'

    def __len__(self):
        return len(self.dataset_samples)

    def augment_sample(self, sample: Dict[str, Union[np.array, str]]) -> Dict[str, Union[np.array, str]]:
        if self.augmentations is None:
            return sample

        sample = copy(sample)

        additional_targets: Dict[str, np.array] = {target_name: sample[target_name]
                              for target_name in self.augmentations.additional_targets.keys()}

        valid_augmentation: bool = False
        aug_output: Dict[str, np.array] = {}
        while not valid_augmentation:
            aug_output = self.augmentations(image=sample['image'],
                                            mask=sample['mask'],
                                            **additional_targets)
            valid_augmentation = self.check_augmented_sample(aug_output)

        for target_name, transformed_target in aug_output.items():
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, aug_output: Dict[str, np.array]) -> bool:
        if self.keep_without_mask > 0. and np.random.rand() < self.keep_without_mask:
            return True
        return aug_output['mask'].sum() > 1.0


class HDataset(BaseDataset):
    def __init__(self, dataset_path: str, split: str, **kwargs) -> None:
        super(HDataset, self).__init__(**kwargs)

        self.dataset_path: Path = Path(dataset_path)
        self._split: str = split
        self._real_images_path = self.dataset_path / 'real_images'
        self._composite_images_path = self.dataset_path / 'composite_images'
        self._masks_path = self.dataset_path / 'masks'

        images_lists_paths = [x for x in self.dataset_path.glob('*.txt') if x.stem.endswith(split)]
        assert len(images_lists_paths) == 1

        with open(images_lists_paths[0], 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]

    # noinspection PyUnresolvedReferences
    def get_sample(self, idx: int) -> Dict[str, Union[np.array, str]]:
        composite_name: str = self.dataset_samples[idx]
        real_name: str = composite_name.split('_')[0] + '.jpg'
        mask_name: str = '_'.join(composite_name.split('_')[:-1]) + '.png'

        composite_path: str = str(self._composite_images_path / composite_name)
        real_path: str = str(self._real_images_path / real_name)
        mask_path: str = str(self._masks_path / mask_name)

        composite_image: np.array = cv2.imread(composite_path)
        composite_image = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)

        real_image: np.array = cv2.imread(real_path)
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

        mask: np.array = cv2.imread(mask_path)
        mask = mask[:, :, 0].astype(np.float32) / 255.

        out: Dict[str, Union[np.array, str]] = {
            'image': composite_image,
            'mask': mask,
            'target': real_image,

            'composite_path': composite_path,
            'gt_path': real_path,
            'mask_path': mask_path,
            'image_idx': idx
        }
        return out