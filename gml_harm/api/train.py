from catalyst import dl
from typing import Dict, Any

from ..engine.utils import (
    create_trainer,
    get_criterions,
    get_criterions_callbacks,
    get_optimizers,
    get_optimizers_callbacks,
    get_checkpoints_callbacks,
    get_scheduler
)
from ..data.utils import get_loaders


def train_model(model, cfg: Dict[str, Any]):
    trainer = create_trainer(cfg['trainer'])

    opts = get_optimizers(model, cfg['optimizers'])
    opts_callbacks = get_optimizers_callbacks(cfg['optimizers_callbabs'])

    crits = get_criterions(cfg['criterions'])
    crits_callbacks = get_criterions_callbacks(cfg['criterions_callbacks'])

    checkpoints_callbacks = get_checkpoints_callbacks(cfg['checkpoints_callbacks'])

    all_callbacks = {}
    for callbacks_dict in [crits_callbacks, opts_callbacks, checkpoints_callbacks]:
        all_callbacks.update(callbacks_dict)

    if 'schedulers' in cfg:
        scheds = get_scheduler(opts, cfg['schedulers'])
        all_callbacks.update({'schedulers': dl.SchedulerCallback()})
    else:
        scheds = None
    loaders = get_loaders(cfg['data'])

    project_name = cfg['project_name']
    experiment_name = cfg['experiment_name']

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