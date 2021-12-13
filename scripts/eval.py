import argparse
import torch

import pandas as pd
import numpy as np

from json import load
from collections import OrderedDict

from gml_harm.utils import init_experiment
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

def main():
    args = parse_args()

    with open(args.config, 'r') as config_file:
        cfg = load(config_file, object_pairs_hook=OrderedDict)

    model = create_model(cfg['model'])
    model['model'].load_state_dict(torch.load(args.ckpt)['model_model_state_dict'])

    datasets = cfg['data']['test']['datasets']
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