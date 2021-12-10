from typing import Mapping, Any, Dict, List
from catalyst import dl

from ..engine.utils import (
    create_trainer,
    get_metric_callbacks
)
from ..data.utils import get_loaders


def evaluate_model(model, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    trainer = create_trainer(cfg['trainer'])

    metric_callbacks: List[dl.Callback] = get_metric_callbacks(cfg['metric_callbacks'])

    loaders = get_loaders(cfg['data'], only_valid=True)
    # metrics: Dict[str, Any] = \
    trainer.train(
        model=model,
        loaders=loaders,
        valid_loader='valid',
        callbacks=metric_callbacks,
        engine=dl.DeviceEngine('cuda:0'),
        verbose=True,
    )
    # return metrics
    return trainer.loader_metrics


def evaluate_metrics_on_signle(model, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    pass