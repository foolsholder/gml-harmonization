import torch


from catalyst import dl
from copy import copy
from torch.optim import Optimizer
from typing import Dict, Any, Tuple, List, Callable, Union

from .supervised import SupervisedTrainer
from .self_supervised import SelfSupervisedTrainer
from .hvqvae_runner import HVQVAERunner

from catalyst.dl import MetricAggregationCallback
from ..core.callbacks.metric_callbacks import (
    MSECallback,
    PSNRCallback,
    fMSECallback,
    FNMSECallback
)


def get_metric_callbacks(metric_callbacks_cfg: List[Dict[str, str]]) -> Dict[str, dl.Callback]:
    callbacks: Dict[str, dl.Callback] = {}

    # noinspection PyTypeChecker
    possible_callbacks: Dict[str, Callable[[Any], dl.Callback]] = {
        "MSECallback": MSECallback,
        "PSNRCallback": PSNRCallback,
        "FNMSECallback": FNMSECallback,
        "fMSECallback": fMSECallback,
        "MetricAggregationCallback": MetricAggregationCallback
    }

    for metric_idx, callback_dct in enumerate(metric_callbacks_cfg):
        callback_dct = copy(callback_dct)
        type_name = callback_dct.pop('type')

        eval_only = False
        if 'eval_only' in callback_dct:
            eval_only = callback_dct.pop('eval_only')

        train_only = False

        if 'train_only' in callback_dct:
            train_only = callback_dct.pop('train_only')

        assert not (train_only and eval_only)

        callback_type = possible_callbacks[type_name]
        callback = callback_type(**callback_dct)

        if eval_only or train_only:
            loader = 'valid' if eval_only else 'train'
            callback = dl.ControlFlowCallback(callback, loaders=loader)

        callbacks['metric_{}'.format(metric_idx + 1)] = callback

    return callbacks


def get_optimizers(model: Union[Dict[str, torch.nn.Module], torch.nn.Module],
                   optimizers_cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Optimizer]:
    """
    :param model: torch.nn.Module
    :param optimizers_cfg: Dict[opt_name, Dict[param_group, opt_params]]
    :return: Dict[str, Optimizer]
    """
    optimizers: Dict[str, Optimizer] = {}

    model_nn_instance = isinstance(model, torch.nn.Module)
    model_dict_instance = isinstance(model, dict)

    for opt_name, opt_cfg in optimizers_cfg.items():
        opt_typename = opt_cfg.pop('type')
        opt_class = getattr(torch.optim, opt_typename)

        groups = opt_cfg.pop('groups')

        for param_group in groups:
            param_group_name = param_group['params']
            if param_group_name != 'model':
                if model_nn_instance:
                    submodel = getattr(model, param_group_name)
                else:
                    assert model_dict_instance
                    submodel = model[param_group_name]
            elif param_group_name == 'model':
                if not model_nn_instance:
                    submodel = model['model']
                else:
                    submodel = model
            param_group.update(dict(params=submodel.parameters()))

        optimizers[opt_name] = opt_class(groups)

    return optimizers


def create_trainer(trainer_cfg: Dict[str, str]) -> dl.Runner:
    trainer_cfg = copy(trainer_cfg)
    trainer_type_name = trainer_cfg.pop('type')

    trainers = {
        'SupervisedTrainer': SupervisedTrainer,
        'SelfSupervisedTrainer': SelfSupervisedTrainer,
        'HVQVAERunner': HVQVAERunner
    }
    trainer_type = trainers[trainer_type_name]

    return trainer_type(**trainer_cfg)


def get_optimizers_callbacks(optimizers_callbacks_cfg: List[Dict[str, Any]]) -> Dict[str, dl.OptimizerCallback]:
    """
    :param optimizers_callbacks_cfg: Sequence[ optim_callback params... ]
    :return:
    """
    opt_callbacks: Dict[str: dl.OptimizerCallback] = {}
    for opt_idx, params in enumerate(optimizers_callbacks_cfg):
        opt_callback = dl.OptimizerCallback(**params)
        opt_callbacks['optimizer_{}'.format(opt_idx + 1)] = opt_callback
    return opt_callbacks


def get_checkpoints_callbacks(checkpoints_cfg: List[Dict[str, Any]]) -> Dict[str, dl.CheckpointCallback]:
    checkpoint_callbacks: Dict[str: dl.CheckpointCallback] = {}
    for chp_idx, params in enumerate(checkpoints_cfg):
        params = copy(params)
        use = params.pop('use')
        if not use:
            continue
        chp_callback = dl.CheckpointCallback(**params)
        checkpoint_callbacks['optimizer_{}'.format(chp_idx + 1)] = chp_callback
    return checkpoint_callbacks


def get_scheduler(optimizers: Dict[str, Optimizer],
                  schedulers_cfg: Dict[str, Any]
                  ) -> Tuple[Dict[str, Any], Dict[str, dl.SchedulerCallback]]:
    """
    :param optimizers: Dict[str, Optimizer]
    :param schedulers_cfg: Dict[str, LRScheduler]
    :return:
    """
    scheds: Dict[str, Any] = {}
    scheds_callbacks: Dict[str, dl.SchedulerCallback] = {}

    for opt_name, sched_params in schedulers_cfg.items():
        dct_params = copy(sched_params)
        sched_type_name = dct_params.pop('type')
        mode = dct_params.pop('mode')
        sched_type = getattr(torch.optim.lr_scheduler, sched_type_name)
        sched = sched_type(optimizer=optimizers[opt_name], **dct_params)
        scheds.update({opt_name: sched})
        scheds_callbacks[opt_name] = dl.SchedulerCallback(mode=mode, scheduler_key=opt_name)

    return scheds, scheds_callbacks
