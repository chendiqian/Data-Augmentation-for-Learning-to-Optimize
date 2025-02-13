import os
import copy

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

from data.collate_func import collate_pos_pair
from data.dataset import LPDataset
from data.prefetch_generator import BackgroundGenerator
from data.transforms import GCNNorm, RandomDropNode, AugmentWrapper
from data.utils import save_run_config
from models.hetero_gnn import TripartiteHeteroPretrainGNN
from trainer import NTXentPretrainer


def pretrain(args: DictConfig, log_folder_name: str = None, run_id: int = 0):
    # drop node first, then normalize degree
    aug_list = [
                RandomDropNode(args.pretrain.drop_rate),
                ]
    transform = [AugmentWrapper(aug_list)]
    if 'gcn' in args.conv:
        transform.append(GCNNorm())
    transform = Compose(transform)
    train_set = LPDataset(args.datapath, 'train', transform=transform)
    valid_set = LPDataset(args.datapath, 'valid', transform=transform)
    if args.debug:
        train_set = train_set[:20]
        valid_set = valid_set[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(train_set,
                              batch_size=args.pretrain.batchsize,
                              shuffle=True,
                              collate_fn=collate_pos_pair)
    val_loader = DataLoader(valid_set,
                            batch_size=args.pretrain.batchsize,
                            shuffle=False,
                            collate_fn=collate_pos_pair)

    model = TripartiteHeteroPretrainGNN(
        conv=args.conv,
        hid_dim=args.hidden,
        num_encode_layers=args.num_encode_layers,
        num_conv_layers=args.num_conv_layers,
        num_pred_layers=args.pretrain.num_pred_layers,
        num_mlp_layers=args.num_mlp_layers,
        backbone_pred_layers=args.backbone_pred_layers,
        norm=args.norm).to(device)
    best_model = copy.deepcopy(model.encoder.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=args.pretrain.lr, weight_decay=args.pretrain.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='max',
                                                     factor=0.5,
                                                     patience=50,
                                                     min_lr=1.e-5)

    trainer = NTXentPretrainer(args.pretrain.temperature)

    pbar = tqdm(range(args.pretrain.epoch))
    for epoch in pbar:
        train_loss, train_acc = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)
        val_loss, val_acc = trainer.eval(BackgroundGenerator(val_loader, device, 4), model)

        if scheduler is not None:
            scheduler.step(val_acc)

        if trainer.best_val_acc < val_acc:
            trainer.patience = 0
            trainer.best_val_acc = val_acc
            best_model = copy.deepcopy(model.encoder.state_dict())
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


@hydra.main(version_base=None, config_path='./config', config_name="pre_fine")
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
