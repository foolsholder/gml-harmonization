import torch


from catalyst import dl
from copy import copy
from torch.optim import Optimizer
from typing import Dict, Any, Sequence, List

from .supervised import SupervisedTrainer
from .self_supervised import SelfSupervisedTrainer

from catalyst.dl import MetricAggregationCallback
from ..core.callbacks.metric_callbacks import (
    MSECallback,
    PSNRCallback,
    FNMSECallback
)


def get_metric_callbacks(metric_callbacks_cfg: List[Dict[str, str]]) -> Dict[str, dl.Callback]:
    callbacks: Dict[str, dl.Callback] = {}

    # noinspection PyTypeChecker
    possible_callbacks: Dict[str, dl.Callback] = {
        "MSECallback": MSECallback,
        "PSNRCallback": PSNRCallback,
        "FNMSECallback": FNMSECallback,
        "MetricAggregationCallback": MetricAggregationCallback
    }

    for metric_idx, callback_dct in enumerate(metric_callbacks_cfg):
        callback_dct = copy(callback_dct)
        type_name = callback_dct.pop('type')
        callback_type = getattr(possible_callbacks, type_name)
        callback = callback_type(**callback_dct)

        callbacks['metric_{}'.format(metric_idx + 1)] = callback

    return callbacks


def get_optimizers(model: torch.nn.Module, optimizers_cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Optimizer]:
    """
    :param model: torch.nn.Module
    :param optimizers_cfg: Dict[opt_name, Dict[param_group, opt_params]]
    :return: Dict[str, Optimizer]
    """
    optimizers: Dict[str, Optimizer] = {}
    for opt_name, opt_cfg in optimizers_cfg.items():
        opt_list_args = []
        opt_class = getattr(torch.optim, opt_name)

        for param_group_name, opt_params in opt_cfg.items():
            param_group = getattr(model, param_group_name)
            group_params = {'params': param_group.parameters()}.update(opt_params)
            opt_list_args += [group_params]

        optimizers[opt_name] = opt_class(opt_list_args)

    return optimizers


def create_trainer(trainer_cfg: Dict[str, str]) -> dl.Runner:
    cfg = copy(trainer_cfg)
    trainer_type_name = cfg.pop('type')

    trainers = {
        'SupervisedTrainer': SupervisedTrainer,
        'SelfSupervisedTrainer': SelfSupervisedTrainer
    }
    trainer_type = trainers[trainer_type_name]

    return trainer_type(**cfg)


def get_optimizers_callbacks(optimizers_callbacks_cfg: Sequence[Dict]) -> Dict[str: dl.OptimizerCallback]:
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


def get_scheduler(optimizers: Dict[str, Optimizer], schedulers_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    :param optimizers: Dict[str, Optimizer]
    :param schedulers_cfg: Dict[str, LRScheduler]
    :return:
    """
    scheds: Dict[str, Any] = {}

    for opt_name, sched_params in schedulers_cfg.items():
        dct_params = copy(sched_params)
        sched_type_name = dct_params.pop('type')
        sched_type = getattr(torch.optim.lr_scheduler, sched_type_name)
        sched = sched_type(optimizer=optimizers[opt_name], **dct_params)
        scheds.update({opt_name: sched})

    return scheds
