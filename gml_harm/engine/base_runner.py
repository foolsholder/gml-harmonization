from abc import ABC

from catalyst import dl

from catalyst.core.callback import Callback
from catalyst.core.logger import ILogger
from catalyst.core.trial import ITrial
from catalyst.engines import IEngine
from catalyst.typing import (
    Criterion,
    Model,
    Optimizer,
    Scheduler,
)
from collections import OrderedDict

from typing import Dict, Union, Optional, OrderedDict as ORDType, List, Any
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.nn.parallel import DistributedDataParallel

from ..data.transforms import ToOriginalScale


class BaseRunner(dl.Runner, ABC):
    def __init__(self, restore_scale=True, sync_bn=True):
        super(BaseRunner, self).__init__()
        self.to_original_scale = None
        self._sync_bn = sync_bn
        if restore_scale is not None:
            self.to_original_scale = ToOriginalScale()

    def get_engine(self) -> dl.IEngine:
        return dl.DistributedDataParallelEngine(sync_bn=self._sync_bn)

    def train(
        self,
        *,
        model: Model,
        raw_datasets: ORDType[str, Dict[str, Dataset]],
        **kwargs
    ) -> None:
        assert isinstance(model, dict)
        self._raw_datasets = raw_datasets
        self.engine = self.get_engine()
        super(BaseRunner, self).train(
            model=model,
            loaders=self.get_loaders('smth'),
            **kwargs
        )

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        loaders = OrderedDict()
        for k, v in self._raw_datasets.items():
            sampler = DistributedSampler(
                v['dataset'],
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=(k == 'train')
            )
            loaders[k] = DataLoader(
                dataset=v['dataset'],
                batch_size=v['batch_size'],
                num_workers=v['num_workers'],
                sampler=sampler
            )
        return loaders
