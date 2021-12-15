from abc import ABC

from catalyst import dl

from collections import OrderedDict

from typing import Dict, OrderedDict as ORDType, Union
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from ..data.transforms import ToOriginalScale


class BaseRunner(dl.Runner, ABC):
    def __init__(self, restore_scale=True, sync_bn=True, seed=1337):
        super(BaseRunner, self).__init__()
        self.to_original_scale = None
        self._sync_bn = sync_bn
        if restore_scale is not None:
            self.to_original_scale = ToOriginalScale()
        self._seed = seed

    def get_engine(self) -> dl.IEngine:
        if self._engine is not None:
            return self._engine
        return dl.DistributedDataParallelEngine(sync_bn=self._sync_bn)

    def train(
        self,
        *,
        raw_datasets: ORDType[str, Dict[str, Dataset]] = None,
        engine: Union["IEngine", str] = None,
        loaders: "OrderedDict[str, DataLoader]" = None,
        **kwargs
    ) -> None:
        # assert isinstance(model, dict)
        self._raw_datasets = raw_datasets
        self._engine = engine if engine is not None else self.get_engine()
        loaders = loaders if loaders is not None else self.get_loaders('smth')
        super(BaseRunner, self).train(
            engine=self._engine,
            loaders=loaders,
            **kwargs
        )

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        if self._loaders is not None:
            return super(BaseRunner, self).get_loaders(stage=stage)
        loaders = OrderedDict()
        for k, v in self._raw_datasets.items():
            sampler = DistributedSampler(
                v['dataset'],
                num_replicas=self._engine.world_size,
                rank=self._engine.rank,
                shuffle=(k == 'train'),
                drop_last=(k == 'train')
            )
            loaders[k] = DataLoader(
                dataset=v['dataset'],
                batch_size=v['batch_size'],
                num_workers=v['num_workers'],
                sampler=sampler,
                pin_memory=True
            )
        return loaders
