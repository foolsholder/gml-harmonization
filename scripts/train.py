import argparse
import torch

from gml_harm.utils import init_experiment
from gml_harm.model.utils import create_model
from gml_harm.api.train import train_model


def main():
    args = parse_args()
    cfg = init_experiment(args)

    model = create_model(cfg['model'])
    train_model(model, cfg)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config.')

    parser.add_argument('--exp_name', type=str, default='', required=True,
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--num_workers', type=int, default=-1,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch_size', type=int, default=-1,
                        help='You can override model batch size by specify positive number.')

    return parser.parse_args()


if __name__ == '__main__':
    main()