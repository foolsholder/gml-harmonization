import os.path

import torch
import torch.distributed as dist

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
    get_scheduler,

)
from ..data.utils import get_loaders, get_raw_datasets


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

    if torch.cuda.device_count() > 1:
        datasets = get_raw_datasets(cfg['data'])
        loaders = None
        engine = None # it's defined in trainer by default
    else:
        datasets = None
        loaders = get_loaders(cfg['data'])
        engine = dl.DeviceEngine()

    trainer.train(
        model=model,
        optimizer=opts,
        scheduler=scheds,
        criterion=crits,
        raw_datasets=datasets,
        loaders=loaders,
        engine=engine,
        valid_loader='valid',
        num_epochs=cfg['num_epochs'],
        callbacks=all_callbacks,
        loggers={
            # "wandb": dl.WandbLogger(project=project_name, name=experiment_name),
            "tensorboard": dl.TensorboardLogger(logdir=experiment_folder),
            "csv": dl.CSVLogger(logdir=experiment_folder),
            # "console": dl.ConsoleLogger(),
            # "mlflow": dl.MLflowLogger(experiment=experiment_name, run=project_name)
        },
        verbose=1,
    )