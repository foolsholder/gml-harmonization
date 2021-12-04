from catalyst import dl
from collections import OrderedDict
from typing import Dict, Any, List

from ..engine.utils import (
    create_trainer,
    get_metric_callbacks,
    get_optimizers,
    get_criterions,
    get_optimizers_callbacks,
    get_checkpoints_callbacks,
    get_scheduler
)
from ..data.utils import get_loaders


def train_model(model, cfg: Dict[str, Any]):
    project_name = cfg['project_name']
    experiment_name = cfg['experiment_name']
    experiment_folder = cfg['experiment_folder']

    trainer = create_trainer(cfg['trainer'])

    opts = get_optimizers(model, cfg['optimizers'])
    opts_callbacks = get_optimizers_callbacks(cfg['optimizers_callbacks'])
    crits = get_criterions(cfg['criterions'])
    metric_callbacks = get_metric_callbacks(cfg['metric_callbacks'])

    checkpoints_callbacks = get_checkpoints_callbacks(
        cfg['checkpoints_callbacks'],
        experiment_folder
    )

    if 'schedulers' in cfg:
        scheds, scheds_callbacks = get_scheduler(opts, cfg['schedulers'])
    else:
        scheds = None
        scheds_callbacks = []
    all_callbacks: List[dl.Callback] = []
    for callbacks_dict in [metric_callbacks, opts_callbacks, scheds_callbacks, checkpoints_callbacks]:
        all_callbacks.extend(callbacks_dict)

    loaders = get_loaders(cfg['data'])

    trainer.train(
        model=model,
        optimizer=opts,
        scheduler=scheds,
        criterion=crits,
        loaders=loaders,
        valid_loader='valid',
        num_epochs=cfg['num_epochs'],
        callbacks=all_callbacks,
        loggers={
            "wandb": dl.WandbLogger(project=project_name, name=experiment_name)
        },
        verbose=1,
    )