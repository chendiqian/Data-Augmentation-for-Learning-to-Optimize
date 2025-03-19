import hydra
import numpy as np
import wandb
from omegaconf import DictConfig

from utils.experiment import save_run_config, setup_wandb
from finetune import finetune
from graphcl_pretrain import pretrain  # it shares the normal infonce pretrain


@hydra.main(version_base=None, config_path='./config', config_name="graphcl")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

    best_val_objgaps = []
    test_objgaps = []

    for run in range(args.exp.runs):
        pretrained_model = pretrain(args, log_folder_name, run)
        val_obj, test_obj = finetune(args, log_folder_name, run, pretrained_model)

        best_val_objgaps.append(val_obj)
        test_objgaps.append(test_obj)

        wandb.log({'test_obj_gap': test_obj})

    wandb.log({
        'best_val_obj_gap': np.mean(best_val_objgaps),
        'test_obj_gap_mean': np.mean(test_objgaps),
        'test_obj_gap_std': np.std(test_objgaps),
    })


if __name__ == '__main__':
    main()
