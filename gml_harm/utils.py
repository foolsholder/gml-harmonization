from json import load, dump
from pathlib import Path
from collections import OrderedDict

def init_experiment(args):
    config_path = args.config

    with open(config_path, 'r') as config_file:
        cfg = load(config_file, object_pairs_hook=OrderedDict)

    cfg['experiment_name'] = args.exp_name
    exp_folder = Path(cfg['experiments_folder']) / args.exp_name
    exp_folder.mkdir(exist_ok=True)

    if args.batch_size != -1:
        cfg['data']['batch_size'] = args.batch_size
    if args.num_workers != -1:
        cfg['data']['num_workers'] = args.num_workers

    with open(exp_folder / 'config.json', 'w') as config_out:
        dump(cfg, config_out)

    return cfg