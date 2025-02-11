import hydra
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from data.utils import save_run_config
from finetune import finetune
from pretrain import pretrain


@hydra.main(version_base=None, config_path='./config', config_name="dropnode")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    best_val_losses = []
    test_losses = []

    for run in range(args.runs):
        pretrained_model = pretrain(args, log_folder_name, run)
        finetuned_model, val_loss, test_loss = finetune(args, log_folder_name, run, pretrained_model)

        best_val_losses.append(val_loss)
        test_losses.append(test_loss)

        wandb.log({'best_val_loss': val_loss, 'test_loss': test_loss})

    wandb.log({
        'best_val_loss': np.mean(best_val_losses),
        'test_loss_mean': np.mean(test_losses),
        'test_loss_std': np.std(test_losses),
    })


if __name__ == '__main__':
    main()
