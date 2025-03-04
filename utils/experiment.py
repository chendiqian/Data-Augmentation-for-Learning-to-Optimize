import os
import time

import torch
import wandb
from omegaconf import DictConfig, OmegaConf


def sync_timer():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def save_run_config(args: DictConfig):
    if args.ckpt:
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        prefix = f'{args.wandb.project}_{args.wandb.name}'
        exist_runs = [d for d in os.listdir('logs') if d.startswith(prefix)]
        log_folder_name = f'logs/{prefix}_exp{len(exist_runs)}'
        os.mkdir(log_folder_name)
        # with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
        #     yaml.dump(vars(args), outfile, default_flow_style=False)
        OmegaConf.save(args, os.path.join(log_folder_name, 'config.yaml'))
        return log_folder_name
    return None


def setup_wandb(args):
    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity
