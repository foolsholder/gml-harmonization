import numpy as np

from typing import Sequence, Any, Dict, Union

from .base_dataset import ABCDataset


class ComposedDataset(ABCDataset):
    def __init__(self, datasets: Sequence[ABCDataset], *args, **kwargs) -> None:
        super(ComposedDataset, self).__init__(*args, **kwargs)
        self._datasets = datasets
        self.dataset_samples = []

        for dataset_idx, dataset in enumerate(datasets):
            self.dataset_samples.extend([
                (dataset_idx, sample_idx) for sample_idx in range(len(dataset))
            ])

    def get_sample(self, idx: int) -> Dict[str, Any]:
        dataset_idx, sample_idx = self.dataset_samples[idx]
        dataset = self._datasets[dataset_idx]
        return dataset.get_sample(sample_idx)

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.dataset_samples[idx]
        dataset = self._datasets[dataset_idx]
        return dataset[sample_idx]

    def augment_sample(self, sample: Dict[str, Union[np.array, str]]) -> Dict[str, Union[np.array, str]]:
        raise "Don't use augmentations in ComposedDataset method"
