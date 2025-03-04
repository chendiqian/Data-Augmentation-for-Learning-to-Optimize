import copy
import os
from typing import List, Dict

import hydra
import torch
import wandb
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

from data.collate_func import collate_pos_pair
from data.dataset import LPDataset
from data.prefetch_generator import BackgroundGenerator
from models.igsd_pretrain_gnn import IGSDPretrainGNN
from trainers.igsd_pretrainer import IGSDPretrainer
from transforms.gcn_norm import GCNNormDumb
from transforms.igsd_ppr_augment import IGSDPageRankAugment
from transforms.wrapper import AnchorAugmentWrapper
from utils.experiment import save_run_config, setup_wandb


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg


def pretrain(args: DictConfig, log_folder_name: str = None, run_id: int = 0):
    aug_list = [IGSDPageRankAugment(args.pretrain.method.IGSDPageRankAugment.strength)]
    transform = [AnchorAugmentWrapper(aug_list)]
    # Don't use GCNnorm during pretraining! It makes the pretraining converge too fast!
    if 'gcn' in args.backbone.conv:
        transform.append(GCNNormDumb())
    transform = Compose(transform)
    train_set = LPDataset(args.datapath, 'train', transform=transform)
    valid_set = LPDataset(args.datapath, 'valid', transform=transform)
    if args.debug:
        train_set = train_set[:20]
        valid_set = valid_set[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collate_fn = collate_pos_pair
    train_loader = DataLoader(train_set,
                              batch_size=args.pretrain.batchsize,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(valid_set,
                            batch_size=args.pretrain.batchsize,
                            shuffle=False,
                            collate_fn=collate_fn)

    model = IGSDPretrainGNN(
        conv=args.backbone.conv,
        hid_dim=args.backbone.hidden,
        num_encode_layers=args.backbone.num_encode_layers,
        num_conv_layers=args.backbone.num_conv_layers,
        num_pred_layers=args.pretrain.num_pred_layers,
        num_mlp_layers=args.backbone.num_mlp_layers,
        backbone_pred_layers=args.backbone.num_pred_layers,
        norm=args.backbone.norm).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.pretrain.lr, weight_decay=args.pretrain.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=50,
                                                     min_lr=1.e-5)

    trainer = IGSDPretrainer(args.pretrain.temperature)

    pbar = tqdm(range(args.pretrain.epochs))

    # the best is the mean of student and teacher nets
    best_model = average_weights([model.online_encoder.state_dict(), model.target_encoder.state_dict()])
    for epoch in pbar:
        train_loss, train_acc = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)
        val_loss, val_acc = trainer.eval(BackgroundGenerator(val_loader, device, 4), model)

        if scheduler is not None:
            scheduler.step(val_loss)

        if trainer.best_val_loss > val_loss:
            trainer.patience = 0
            trainer.best_val_loss = val_loss
            best_model = average_weights([model.online_encoder.state_dict(), model.target_encoder.state_dict()])
            if args.ckpt:
                torch.save(best_model, os.path.join(log_folder_name, f'best_model{run_id}.pt'))
        else:
            trainer.patience += 1

        if trainer.patience > args.pretrain.patience:
            break

        stats_dict = {'pretrain_train_loss': train_loss,
                      'pretrain_train_acc': train_acc,
                      'pretrain_val_loss': val_loss,
                      'pretrain_val_acc': val_acc,
                      'pretrain_lr': scheduler.optimizer.param_groups[0]["lr"]}

        pbar.set_postfix(stats_dict)
        wandb.log(stats_dict)
    return best_model


@hydra.main(version_base=None, config_path='./config', config_name="igsd")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

    for run in range(args.runs):
        pretrain(args, log_folder_name, run)


if __name__ == '__main__':
    main()
