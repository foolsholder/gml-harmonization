import argparse
import torch

import pandas as pd
import numpy as np

from json import load
from collections import OrderedDict

from gml_harm.model.utils import create_model
from gml_harm.api.eval import evaluate_model


def _compute_iHarmony4_metrics(res_full: pd.DataFrame):
    res = res_full[['HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night']]

    total_count_samples = res.loc['count_samples'].sum()
    iHarmony4_col = []

    for idx in res.index:
        if idx == 'count_samples':
            iHarmony4_col += [total_count_samples]
            continue
        iHarmony4_col += [(
                              (res.loc[idx] * res.loc['count_samples']).sum()
                          ) / total_count_samples]
    res_full['iHarmony4'] = iHarmony4_col
    return res_full


def bn_update(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be trasferred to
            :attr:`device` before being passed into :attr:`model`.
    """
    if not _check_bn(model):
        return
    was_training = model.training
    model.train()
    momenta = {}
    model.to(device)
    model.apply(_reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    base = getattr(model, 'base')
    mask_fusion_bb = getattr(model, 'mask_fusion_bb')
    n = 0
    from tqdm.auto import tqdm
    for _ in range(2):
        for input in tqdm(loader):
            images = input['images']
            masks = input['masks']
            b = images.size(0)

            momentum = b / float(n + b)
            for module in momenta.keys():
                module.momentum = momentum

            if device is not None:
                images = images.to(device)
                masks = masks.to(device)

            with torch.no_grad():
                model(images, masks)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    model.train(was_training)


# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def experiment_bn(model, cfg):
    from gml_harm.data.utils import get_loaders
    cfg['data']['train']['augmentations'] = cfg['data']['test']['augmentations']
    cfg['data']['test']['augmentations'].update(
        {}
    )
    cfg['data']['batch_size'] = 120
    loaders = get_loaders(cfg['data'])
    train_loader = loaders['train']
    bn_update(train_loader, model['model'], device='cuda:0')
    model['model'].eval()


def main():
    args = parse_args()

    with open(args.config, 'r') as config_file:
        cfg = load(config_file, object_pairs_hook=OrderedDict)

    if args.batch_size != -1:
        cfg['data']['batch_size'] = args.batch_size
    if args.num_workers != -1:
        cfg['data']['num_workers'] = args.num_workers

    model = create_model(cfg['model'])
    model['model'].load_state_dict(torch.load(args.ckpt)['model_model_state_dict'])

    if False:
        experiment_bn(model, cfg)

    datasets = [
        'HCOCO',
        'HAdobe5k',
        'HFlickr',
        'Hday2night',
        #'RealHM',
        #'HVIDIT'
    ]
    res = {}
    for dataset in datasets:
        cfg['data']['test']['datasets'] = [dataset]
        metrics = evaluate_model(model, cfg)
        res[dataset] = metrics
    res = pd.DataFrame(res)

    res = _compute_iHarmony4_metrics(res)

    res.to_csv(cfg['experiment_folder'] + '/eval_res.csv', index_label='metric')
    pd.set_option('display.max_rows', res.shape[0] + 1)
    print(res)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config.')

    parser.add_argument('--ckpt', type=str, required=True)

    parser.add_argument('--num_workers', type=int, default=-1,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch_size', type=int, default=-1,
                        help='You can override model batch size by specify positive number.')

    return parser.parse_args()


if __name__ == '__main__':
    main()