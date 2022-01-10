import argparse
import torch

from gml_harm.utils import init_experiment
from gml_harm.model.utils import create_model
from gml_harm.api.train import train_model
from gml_harm.api.extra import train_swa


def main():
    args = parse_args()
    cfg = init_experiment(args)

    model = create_model(cfg['model'])

    if args.resume:
        cfg['resume'] = args.resume

    if args.swa:
        train_swa(model, cfg)
    else:
        train_model(model, cfg)


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

    parser.add_argument('--seed', type=int, default=-1,
                        help='You can override default random seed=1337 for training')
    parser.add_argument('--num_epochs', type=int, default=-1)
    parser.add_argument('--without_ckpt', type=bool, default=False)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--swa', type=bool, default=False)

    return parser.parse_args()


if __name__ == '__main__':
    main()