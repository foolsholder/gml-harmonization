import argparse
import torch
import os
import json

from gml_harm.utils import init_experiment
from gml_harm.model.utils import create_model
from gml_harm.api.train import train_model
from gml_harm.api.eval import evaluate_model


def main():
    args = parse_args()

    args.seeds = list(map(lambda x: int(x), args.seeds.split(',')))
    args.without_ckpt = True

    exp_name = args.exp_name

    for seed in args.seeds:
        args.seed = seed
        args.exp_name = exp_name + f'_seed{seed}'
        cfg = init_experiment(args)

        model = create_model(cfg['model'])
        train_model(model, cfg)
        metrics = evaluate_model(model, cfg)
        with open(os.path.join(cfg['experiment_folder'], 'eval_metrics.json'), 'w') as out:
            json.dump(metrics, out, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config.')

    parser.add_argument('--exp_name', type=str, default='', required=True,
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--project_name', type=str, default='')

    parser.add_argument('--num_workers', type=int, default=-1,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch_size', type=int, default=-1,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--seeds', type=str, default='37,1337,131')
    parser.add_argument('--num_epochs', type=int, default=-1)

    return parser.parse_args()


if __name__ == '__main__':
    main()