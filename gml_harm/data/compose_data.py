from typing import Sequence, Any, Dict

from .base_dataset import BaseDataset


class ComposedDataset(BaseDataset):
    def __init__(self, datasets: Sequence[BaseDataset], *args, **kwargs) -> None:
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