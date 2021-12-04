import torch

from catalyst import dl
from collections import OrderedDict
from copy import copy
from torch.optim import Optimizer
from typing import (
    Dict,
    Any,
    Tuple,
    List,
    Callable,
    Union,
    OrderedDict as ORDType
)

from .supervised import SupervisedTrainer
from .self_supervised import SelfSupervisedTrainer
from .hvqvae_runner import HVQVAERunner

from catalyst.dl import MetricAggregationCallback, CriterionCallback
from ..core.callbacks.metric_callbacks import (
    MSECallback,
    PSNRCallback,
    fMSECallback,
    FNMSECallback,
    IdentityCallback
)

from ..core.callbacks.optimizer_callbacks import (
    OptimizerCallback
)

from ..core.criterions import (
    MSECriterion,
    PSNRCriterion,
    fMSECriterion,
    FNMSECriterion
)


def get_criterions(
        criterion_cfg: ORDType[str, ORDType[str, str]]
    ) -> ORDType[str, torch.nn.Module]:
    possible_criterions: Dict[str, torch.nn.Module] = {
        "MSECriterion": MSECriterion,
        "PSNRCriterion": PSNRCriterion,
        "fMSECriterion": fMSECriterion,
        "FNMSECriterion": FNMSECriterion
    }
    crits: ORDType[str, torch.nn.Module] = OrderedDict()
    for crit_name, crit_params in criterion_cfg.items():
        crit_typename = crit_params.pop('type')
        crit_type = possible_criterions[crit_typename]
        crits[crit_name] = crit_type(**crit_params)
    return crits


def get_metric_callbacks(
        metric_callbacks_cfg: List[ORDType[str, str]]
    ) -> List[dl.Callback]:
    callbacks: List[dl.Callback] = []

    # noinspection PyTypeChecker
    possible_callbacks: Dict[str, Callable[[Any], dl.Callback]] = {
        "CriterionCallback": CriterionCallback,
        "MSECallback": MSECallback,
        "PSNRCallback": PSNRCallback,
        "FNMSECallback": FNMSECallback,
        "fMSECallback": fMSECallback,
        "IdentityCallback": IdentityCallback,
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

        callbacks += [callback]

    return callbacks


def get_optimizers(
        model: Union[ORDType[str, torch.nn.Module], torch.nn.Module],
        optimizers_cfg: ORDType[str, Dict[str, Any]]
    ) -> ORDType[str, Optimizer]:
    """
    :param model: torch.nn.Module
    :param optimizers_cfg: Dict[opt_name, Dict[param_group, opt_params]]
    :return: OrderedDict[str, Optimizer]
    """
    optimizers: ORDType[str, Optimizer] = OrderedDict()

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


def create_trainer(trainer_cfg: ORDType[str, str]) -> dl.Runner:
    trainer_cfg = copy(trainer_cfg)
    trainer_type_name = trainer_cfg.pop('type')

    trainers = {
        'SupervisedTrainer': SupervisedTrainer,
        'SelfSupervisedTrainer': SelfSupervisedTrainer,
        'HVQVAERunner': HVQVAERunner
    }
    trainer_type = trainers[trainer_type_name]

    return trainer_type(**trainer_cfg)


def get_optimizers_callbacks(
        optimizers_callbacks_cfg: List[ORDType[str, Any]]
    ) -> List[OptimizerCallback]:
    """
    :param optimizers_callbacks_cfg: Sequence[ optim_callback params... ]
    :return:
    """
    opt_callbacks: List[OptimizerCallback] = []
    for opt_idx, params in enumerate(optimizers_callbacks_cfg):
        opt_callback = OptimizerCallback(**params)
        opt_callbacks += [opt_callback]
    return opt_callbacks


def get_checkpoints_callbacks(
        checkpoints_cfg: List[ORDType[str, Any]],
        experiment_folder: str) -> List[dl.CheckpointCallback]:
    checkpoint_callbacks: List[dl.CheckpointCallback] = []
    for chp_idx, params in enumerate(checkpoints_cfg):
        params = copy(params)
        use = params.pop('use')
        if not use:
            continue

        params['logdir'] = experiment_folder  + '/' + 'checkpoints'
        chp_callback = dl.CheckpointCallback(**params)
        checkpoint_callbacks += [chp_callback]
    return checkpoint_callbacks


def get_scheduler(
        optimizers: ORDType[str, Optimizer],
        schedulers_cfg: ORDType[str, Any]
    ) -> Tuple[ORDType[str, Any], List[dl.SchedulerCallback]]:
    """
    :param optimizers: Dict[str, Optimizer]
    :param schedulers_cfg: Dict[str, LRScheduler]
    :return:
    """
    scheds: ORDType[str, Any] = OrderedDict()
    scheds_callbacks: List[dl.SchedulerCallback] = []

    for opt_name, sched_params in schedulers_cfg.items():
        dct_params = copy(sched_params)
        sched_type_name = dct_params.pop('type')
        mode = dct_params.pop('mode')
        sched_type = getattr(torch.optim.lr_scheduler, sched_type_name)
        sched = sched_type(optimizer=optimizers[opt_name], **dct_params)
        scheds.update({opt_name: sched})
        scheds_callbacks += [dl.SchedulerCallback(mode=mode, scheduler_key=opt_name)]

    return scheds, scheds_callbacks
