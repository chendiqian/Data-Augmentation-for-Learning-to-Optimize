import os

import hydra
import copy
import numpy as np
import torch
from torch import optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from torch_geometric.datasets import ZINC
from data.transforms import GCNNorm
from data.prefetch_generator import BackgroundGenerator
from mol_models.gnn import BasicGNN
from trainer import PlainGNNTrainer
from data.utils import save_run_config


@hydra.main(version_base=None, config_path='./config', config_name="run")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    transform = GCNNorm() if 'gcn' in args.conv else None
    train_set = ZINC('./datasets', subset=True, split='train', transform=transform)
    if args.train_frac < 1:
        train_set = train_set[:int(len(train_set) * args.train_frac)]
    valid_set = ZINC('./datasets', subset=True, split='val', transform=transform)
    test_set = ZINC('./datasets', subset=True, split='test', transform=transform)
    if args.debug:
        train_set = train_set[:20]
        valid_set = valid_set[:20]
        test_set = test_set[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(train_set,
                              batch_size=args.batchsize,
                              shuffle=True)
    val_loader = DataLoader(valid_set,
                            batch_size=args.batchsize,
                            shuffle=False)
    test_loader = DataLoader(test_set,
                             batch_size=args.batchsize,
                             shuffle=False)

    best_val_loss = []
    test_losses = []

    for run in range(args.runs):
        model = BasicGNN(conv=args.conv,
                         hid_dim=args.hidden,
                         num_conv_layers=args.num_conv_layers,
                         num_pred_layers=args.num_pred_layers,
                         num_mlp_layers=args.num_mlp_layers,
                         norm=args.norm).to(device)
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=50,
                                                         min_lr=1.e-5)

        trainer = PlainGNNTrainer(args.losstype)

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)
            val_loss = trainer.eval(BackgroundGenerator(val_loader, device, 4), model)

            if scheduler is not None:
                scheduler.step(val_loss)

            if trainer.best_val_loss > val_loss:
                trainer.patience = 0
                trainer.best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                if args.ckpt:
                    torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
            else:
                trainer.patience += 1

            if trainer.patience > args.patience:
                break

            stats_dict = {'train_loss': train_loss,
                          'val_loss': val_loss,
                          'lr': scheduler.optimizer.param_groups[0]["lr"]}

            pbar.set_postfix(stats_dict)
            wandb.log(stats_dict)
        best_val_loss.append(trainer.best_val_loss)

        model.load_state_dict(best_model)
        test_loss = trainer.eval(test_loader, model)
        test_losses.append(test_loss)
        wandb.log({'test_obj_gap': test_loss})

    wandb.log({
        'best_val_loss': np.mean(best_val_loss),
        'test_obj_loss_mean': np.mean(test_losses),
        'test_obj_loss_std': np.std(test_losses)
    })


if __name__ == '__main__':
    main()
