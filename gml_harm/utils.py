from json import load, dump
from pathlib import Path
from collections import OrderedDict


def _disable_ckpt(cfg):
    for ckpt_callback in cfg['checkpoints_callbacks']:
        ckpt_callback['use'] = False


def init_experiment(args):
    config_path = args.config

    with open(config_path, 'r') as config_file:
        cfg = load(config_file, object_pairs_hook=OrderedDict)

    if args.project_name:
        cfg['project_name'] = args.project_name

    if args.seed != -1:
        cfg['trainer']['seed'] = args.seed

    if args.num_epochs != -1:
        cfg['num_epochs'] = args.num_epochs

    cfg['experiment_name'] = args.exp_name
    experiment_folder = Path(cfg['experiments_folder'] + '/' + args.exp_name)
    cfg['experiment_folder'] = str(experiment_folder)
    exp_folder = experiment_folder
    exp_folder.mkdir(exist_ok=True)

    if args.batch_size != -1:
        cfg['data']['batch_size'] = args.batch_size
    if args.num_workers != -1:
        cfg['data']['num_workers'] = args.num_workers

    if args.without_ckpt:
        _disable_ckpt(cfg)

    with open(exp_folder / 'config.json', 'w') as config_out:
        dump(cfg, config_out, indent=2)

    return cfg
