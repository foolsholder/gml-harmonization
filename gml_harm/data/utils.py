from albumentations import (
    BasicTransform,
    Compose,
    Normalize,
    Resize,
    RandomResizedCrop,
    HorizontalFlip
)
from albumentations.pytorch import ToTensorV2
from copy import copy
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, List,Type, Tuple, Union

from .base_dataset import HDataset, ABCDataset
from .compose_data import ComposedDataset
from .ssh_dataset import SSHTrainDataset

def get_ssh_crop(crop_cfg: Dict[str, Union[float, int]]) -> Compose:
    available_crop: Dict[str, Type[BasicTransform]] = {
        'RandomResizedCrop': RandomResizedCrop
    }
    crop_typename = crop_cfg.pop('type')
    crop_type = available_crop[crop_typename]
    crop = Compose(
        [crop_type(**crop_cfg)],
        p=1.0,
        additional_targets={
            'reference_alpha': 'image',
            'content_beta': 'image',
            'reference_beta': 'image'
        }
    )
    return crop

def get_transform(augmentations_cfg: Dict[str, Dict[str, str]]) -> Tuple[Compose, Compose]:
    """
    creates simple Sequential Composition of augmentations
    doesn't provide any opportunities to use Compose([..., Compose(...), ...])
    """
    to_tensor_transforms: Compose = Compose(
        [
            Normalize(),
            ToTensorV2()
        ],
        p=1.0,
        additional_targets={
            "target": "image"
        }
    )
    augmentations_dict = {
        "Resize": Resize,
        "RandomResizedCrop": RandomResizedCrop,
        "HorizontalFlip": HorizontalFlip
    }
    augmentations_list: List[BasicTransform] = []
    for aug_type_name, aug_params in augmentations_cfg.items():
        aug_type = augmentations_dict[aug_type_name]
        augmentations_list += [aug_type(**aug_params)]
    augmentations = Compose(
        augmentations_list,
        p=1.0,
        additional_targets={
            "target": "image"
        }
    )
    return augmentations, to_tensor_transforms


def get_dataset(data_cfg: Dict[str, Any], split: str) -> Dataset:
    available_datasets: Dict[str, Type[ABCDataset]] = {
        'HDataset': HDataset,
        'SSHTrainDataset': SSHTrainDataset
    }
    dataset_cfg = data_cfg[split]
    dataset_cfg = copy(dataset_cfg)

    datasets = dataset_cfg.pop('datasets')

    dataset_typename = dataset_cfg.pop('type')
    dataset_type = available_datasets[dataset_typename]

    augmentations_cfg = dataset_cfg.pop('augmentations')
    augmentations, to_tensor_transforms = get_transform(augmentations_cfg)

    dataset_paths = data_cfg['dataset_paths']

    dataset_cfg['augmentations'] = augmentations
    dataset_cfg['to_tensor_transforms'] = to_tensor_transforms
    dataset_cfg['split'] = split
    if 'crop' in dataset_cfg:
        dataset_cfg['crop'] = get_ssh_crop(dataset_cfg['crop'])

    datasets: List[ABCDataset] = []
    for ds_name in datasets:
        datasets.append(dataset_type.__init__(
            dataset_paths[ds_name], **dataset_cfg
        ))
    return ComposedDataset(datasets)


def get_dataloader(data_cfg: Dict[str, Any], train: bool = False) -> DataLoader:
    split = 'train' if train else 'test'
    dataset: Dataset = get_dataset(data_cfg, split)

    num_workers: int = data_cfg['num_workers']
    batch_size: int = data_cfg['batch_size']
    data_loader = DataLoader(dataset,
                             num_workers=num_workers,
                             batch_size=batch_size,
                             shuffle=train,
                             drop_last=train)
    return data_loader


def get_loaders(data_cfg: Dict[str, Any], only_valid: bool = False) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    if not only_valid:
        loaders.update({
            'train': get_dataloader(data_cfg, train=True)
        })
    loaders.update({
        'valid': get_dataloader(data_cfg, train=False)
    })

    return loaders
