import torch.utils.data
from albumentations import (
    BasicTransform,
    Compose,
    Normalize,
    Resize,
    RandomResizedCrop,
    RandomCrop,
    HorizontalFlip,
    RGBShift,
    CLAHE,
    HueSaturationValue,
    RandomBrightnessContrast,
    OpticalDistortion,
    PadIfNeeded,
    Rotate
)

from albumentations.pytorch import ToTensorV2
from collections import OrderedDict
from copy import copy, deepcopy
from torch.utils.data import DataLoader, Dataset
from typing import (
    Dict,
    Any,
    List,
    Type,
    Tuple,
    Union,
    OrderedDict as ORDType,
    Optional
)

from .base_dataset import HDataset, ABCDataset
from .compose_data import ComposedDataset
from .lut_dataset import LutDataset
from .ssh_dataset import SSHTrainDataset, SSHTestDataset
from .transforms import IAAAffine2, IAAPerspective2, LookUpTableAUG


def get_single_crop(
        crop_cfg: ORDType[str, Union[float, int]],
        additional_targets: Optional[Dict[str, str]] = None) -> Compose:
    available_crop: Dict[str, Type[BasicTransform]] = {
        'RandomResizedCrop': RandomResizedCrop,
        'RandomCrop': RandomCrop
    }
    crop_typename = crop_cfg.pop('type')
    crop_type = available_crop[crop_typename]

    padd = False
    if 'PadIfNeeded' in crop_cfg:
        padd = crop_cfg.pop('PadIfNeeded')

    crop_list: List[BasicTransform] = [crop_type(**crop_cfg)]

    if padd:
        crop_list = [PadIfNeeded(min_height=crop_cfg['height'], min_width=crop_cfg['width'])] + crop_list

    if additional_targets is None:
        additional_targets = {
            'view_beta': 'image'
        }

    crop = Compose(
        crop_list,
        p=1.0,
        additional_targets=additional_targets
    )

    return crop


def get_augmentations(
        augmentations_cfg: ORDType[str, Dict[str, Union[str, float, int]]],
        additional_targets: Dict[str, str]
    ) -> Compose:
    """
    creates simple Sequential Composition of augmentations
    doesn't provide any opportunities to use Compose([..., Compose(...), ...])
    """
    augmentations_dict = {
        "Rotate": Rotate,
        "Resize": Resize,
        "RandomResizedCrop": RandomResizedCrop,
        "HorizontalFlip": HorizontalFlip,
        "RGBShift": RGBShift,
        "CLAHE": CLAHE,
        "HueSaturationValue": HueSaturationValue,
        "RandomBrightnessContrast": RandomBrightnessContrast,
        "OpticalDistortion": OpticalDistortion,
        "IAAAffine2": IAAAffine2,
        "IAAPerspective2": IAAPerspective2,
        "PadIfNeeded": PadIfNeeded,
        "Normalize": Normalize,
        "ToTensorV2": ToTensorV2,
        "LookUpTableAUG": LookUpTableAUG
    }
    augmentations_list: List[BasicTransform] = []
    for aug_type_name, aug_params in augmentations_cfg.items():
        aug_type = augmentations_dict[aug_type_name]
        augmentations_list += [aug_type(**aug_params)]
    augmentations = Compose(
        augmentations_list,
        p=1.0,
        additional_targets=additional_targets
    )
    return augmentations


def get_dataset(data_cfg: ORDType[str, Any], split: str) -> Dataset:
    available_datasets: Dict[str, Type[ABCDataset]] = {
        'HDataset': HDataset,
        'SSHTrainDataset': SSHTrainDataset,
        'SSHTestDataset': SSHTestDataset,
        'LutDataset': LutDataset
    }
    dataset_cfg = data_cfg[split]
    dataset_cfg = deepcopy(dataset_cfg)

    datasets = dataset_cfg.pop('datasets')

    dataset_typename = dataset_cfg.pop('type')
    dataset_type = available_datasets[dataset_typename]

    additional_targets: Dict[str, str] = dataset_cfg.pop('additional_targets')

    augmentations_dict: Dict[str, Compose] = {}

    aug_components = ['to_tensor_transforms', 'augmentations', 'geometric_augmentations', 'color_augmentations']
    for aug_name in aug_components:
        if aug_name in dataset_cfg:
            augmentations_cfg = dataset_cfg.pop(aug_name)
            augmentations_dict[aug_name] = get_augmentations(augmentations_cfg, additional_targets)

    dataset_paths = data_cfg['dataset_paths']

    dataset_cfg.update(augmentations_dict)
    dataset_cfg['split'] = split
    if 'crop' in dataset_cfg:
        dataset_cfg['crop'] = get_single_crop(dataset_cfg['crop'], additional_targets)

    builed_datasets: List[ABCDataset] = []
    for ds_name in datasets:
        builed_datasets.append(dataset_type(
            dataset_path=dataset_paths[ds_name], **dataset_cfg
        ))
    return ComposedDataset(builed_datasets)


def get_raw_dataset(data_cfg: ORDType[str, Any],
                    train: bool = False) -> Dict[str, Dataset]:
    split = 'train' if train else 'test'
    dataset: Dataset = get_dataset(data_cfg, split)

    num_workers: int = data_cfg['num_workers']
    batch_size: int = data_cfg['batch_size']
    return {
        'split': split,
        'dataset': dataset,
        'num_workers': num_workers,
        'batch_size': batch_size
    }


def get_raw_datasets(data_cfg: ORDType[str, Any],
                     only_valid: bool = False) -> ORDType[str, Dict[str, Dataset]]:
    res = OrderedDict()
    if not only_valid:
        res.update({'train': get_raw_dataset(data_cfg, train=True)})
    res.update({'valid': get_raw_dataset(data_cfg, train=False)})
    return res


def get_dataloader(data_cfg: ORDType[str, Any], train: bool = False) -> DataLoader:
    split = 'train' if train else 'test'
    dataset: Dataset = get_dataset(data_cfg, split)

    num_workers: int = data_cfg['num_workers']
    batch_size: int = data_cfg['batch_size']
    data_loader = DataLoader(dataset,
                             num_workers=num_workers,
                             batch_size=batch_size,
                             shuffle=train,
                             drop_last=train,
                             pin_memory=True)
    return data_loader


def get_loaders(data_cfg: ORDType[str, Any],
                only_valid: bool = False) -> ORDType[str, DataLoader]:
    loaders: ORDType[str, DataLoader] = OrderedDict()
    if not only_valid:
        loaders.update({
            'train': get_dataloader(data_cfg, train=True)
        })
    loaders.update({
        'valid': get_dataloader(data_cfg, train=False)
    })

    return loaders
