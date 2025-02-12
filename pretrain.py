import os

import copy

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
from torch_geometric.datasets import ZINC
from data.prefetch_generator import BackgroundGenerator
from data.transforms import (GCNNorm,
                             RandomDropNode,
                             RandomMaskNodeAttr,
                             RandomDropEdge,
                             IdentityAugmentation,
                             AugmentWrapper)
from data.utils import save_run_config
from mol_models.encoder import Encoder
from trainer import NTXentPretrainer


def pretrain(args: DictConfig, log_folder_name: str = None, run_id: int = 0):
    # drop node first, then normalize degree
    aug_list = [RandomMaskNodeAttr(args.pretrain.drop_rate),
                RandomDropNode(args.pretrain.drop_rate),
                RandomDropEdge(args.pretrain.drop_rate),
                # IdentityAugmentation(),
                ]
    transform = [AugmentWrapper(aug_list)]
    if 'gcn' in args.conv:
        transform.append(GCNNorm())
    transform = Compose(transform)
    pretrain_set = ZINC('./datasets', subset=args.pretrain.subset, split='train', transform=transform)
    prevalid_set = ZINC('./datasets', subset=True, split='val', transform=transform)

    if args.debug:
        pretrain_set = pretrain_set[:20]
        prevalid_set = prevalid_set[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrain_loader = DataLoader(pretrain_set,
                                 batch_size=args.pretrain.batchsize,
                                 shuffle=True)
    preval_loader = DataLoader(prevalid_set,
                               batch_size=args.pretrain.batchsize,
                               shuffle=False)

    model = Encoder(conv=args.conv,
                    hid_dim=args.hidden,
                    num_conv_layers=args.num_conv_layers,
                    num_pred_layers=args.num_pred_layers,
                    num_mlp_layers=args.num_mlp_layers,
                    norm=args.norm).to(device)
    best_model = copy.deepcopy(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=args.pretrain.lr, weight_decay=args.pretrain.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='max',
                                                     factor=0.5,
                                                     patience=50,
                                                     min_lr=1.e-5)

    trainer = NTXentPretrainer(args.pretrain.temperature)

    pbar = tqdm(range(args.pretrain.epoch))
    for epoch in pbar:
        train_loss, train_acc = trainer.train(BackgroundGenerator(pretrain_loader, device, 4), model, optimizer)
        val_loss, val_acc = trainer.eval(BackgroundGenerator(preval_loader, device, 4), model)

        if scheduler is not None:
            scheduler.step(val_acc)

        if trainer.best_val_acc < val_acc:
            trainer.patience = 0
            trainer.best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            if args.ckpt:
                torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run_id}.pt'))
        else:
            trainer.patience += 1

        if trainer.patience > args.pretrain.patience:
            break

        stats_dict = {'pretrain_train_loss': train_loss,
                      'pretrain_train_top5acc': train_acc,
                      'pretrain_val_loss': val_loss,
                      'pretrain_val_top5acc': val_acc,
                      'pretrain_lr': scheduler.optimizer.param_groups[0]["lr"]}

        pbar.set_postfix(stats_dict)
        wandb.log(stats_dict)

    return best_model


@hydra.main(version_base=None, config_path='./config', config_name="pretrain_finetune")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    for run in range(args.runs):
        pretrain(args, log_folder_name, run)


if __name__ == '__main__':
    main()
