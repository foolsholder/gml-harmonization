from typing import Mapping, Any

from ..engine.utils import (
    create_trainer,
    get_criterions,
    get_criterions_callbacks
)
from ..data.utils import get_loaders


def evaluate_model(model, cfg: Mapping[str, Any]):
    trainer = create_trainer(cfg['trainer'])

    crits = get_criterions(cfg['criterions'])
    crits_callbacks = get_criterions_callbacks(cfg['criterions_callbacks'])

    loaders = get_loaders(cfg['data'])
    trainer.train(
        model=model,
        criterion=crits,
        loaders=loaders,
        valid_loader='valid',
        num_epochs=1,
        callbacks=crits_callbacks,
        verbose=1,
    )