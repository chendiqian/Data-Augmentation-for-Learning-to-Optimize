import hydra
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from data.utils import save_run_config
from finetune import finetune
from infograph_pretrain import pretrain


@hydra.main(version_base=None, config_path='./config', config_name="infograph")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    best_val_objgaps = []
    test_objgaps = []

    for run in range(args.runs):
        pretrained_model = pretrain(args, log_folder_name, run)
        finetuned_model, val_obj, test_obj = finetune(args, log_folder_name, run, pretrained_model)

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
