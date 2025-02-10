import os

import copy
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset, DataLoader as PTDataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
import numpy as np
from torch_geometric.datasets import ZINC
from data.prefetch_generator import BackgroundGenerator
from data.transforms import GCNNorm, RandomDropNode
from data.utils import save_run_config
from mol_models.encoder import Encoder
from trainer import NTXentPretrainer, LinearTrainer


@hydra.main(version_base=None, config_path='./config', config_name="dropnode")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    # drop node first, then normalize degree
    transform = [RandomDropNode(args.drop_rate)]
    if 'gcn' in args.conv:
        transform.append(GCNNorm())
    transform = Compose(transform)
    pretrain_set = ZINC('./datasets', subset=True, split='train', transform=transform)
    prevalid_set = ZINC('./datasets', subset=True, split='val', transform=transform)

    transform = GCNNorm() if 'gcn' in args.conv else None
    finetune_train_set = ZINC('./datasets', subset=True, split='train', transform=transform)
    if args.train_frac < 1:
        finetune_train_set = finetune_train_set[:int(len(finetune_train_set) * args.train_frac)]
    finetune_valid_set = ZINC('./datasets', subset=True, split='val', transform=transform)
    finetune_test_set = ZINC('./datasets', subset=True, split='test', transform=transform)
    if args.debug:
        pretrain_set = pretrain_set[:20]
        prevalid_set = prevalid_set[:20]
        finetune_train_set = finetune_train_set[:20]
        finetune_valid_set = finetune_valid_set[:20]
        finetune_test_set = finetune_test_set[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrain_loader = DataLoader(pretrain_set,
                              batch_size=args.batchsize,
                              shuffle=True)
    preval_loader = DataLoader(prevalid_set,
                            batch_size=args.batchsize,
                            shuffle=False)

    finetune_train_loader = DataLoader(finetune_train_set,
                              batch_size=args.batchsize,
                              shuffle=False)
    finetune_val_loader = DataLoader(finetune_valid_set,
                            batch_size=args.batchsize,
                            shuffle=False)
    finetune_test_loader = DataLoader(finetune_test_set,
                             batch_size=args.batchsize,
                             shuffle=False)

    best_val_losses = []
    test_losses = []

    for run in range(args.runs):
        model = Encoder(conv=args.conv,
                        hid_dim=args.hidden,
                        num_conv_layers=args.num_conv_layers,
                        num_pred_layers=args.num_pred_layers,
                        num_mlp_layers=args.num_mlp_layers,
                        norm=args.norm).to(device)
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='max',
                                                         factor=0.5,
                                                         patience=50 // args.eval_every,
                                                         min_lr=1.e-5)

        trainer = NTXentPretrainer(args.temperature)

        pbar = tqdm(range(args.pretrain_epoch))
        for epoch in pbar:
            train_loss, train_acc = trainer.train(BackgroundGenerator(pretrain_loader, device, 4), model, optimizer)
            stats_dict = {'pretrain_train_loss': train_loss,
                          'pretrain_train_top5acc': train_acc,
                          'pretrain_lr': scheduler.optimizer.param_groups[0]["lr"]}
            if epoch % args.eval_every == 0:
                val_loss, val_acc = trainer.eval(BackgroundGenerator(preval_loader, device, 4), model)

                if scheduler is not None:
                    scheduler.step(val_acc)

                if trainer.best_val_acc < val_acc:
                    trainer.patience = 0
                    trainer.best_val_acc = val_acc
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt:
                        torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
                else:
                    trainer.patience += 1

                if trainer.patience > (args.pretrain_patience // args.eval_every + 1):
                    break

                stats_dict['pretrain_val_loss'] = val_loss
                stats_dict['pretrain_val_top5acc'] = val_acc

            pbar.set_postfix(stats_dict)
            wandb.log(stats_dict)

        model.load_state_dict(best_model)

        def get_feat_label(loader):
            model.eval()
            features = []
            labels = []
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    obj_pred = model(data)
                    label = data.y
                    features.append(obj_pred)
                    labels.append(label)
            features = torch.cat(features, dim=0)
            labels = torch.cat(labels, dim=0)
            return TensorDataset(features, labels)

        train_ds = get_feat_label(finetune_train_loader)
        val_ds = get_feat_label(finetune_val_loader)
        test_ds = get_feat_label(finetune_test_loader)

        train_loader = PTDataLoader(train_ds, batch_size=args.batchsize, shuffle=True)
        val_loader = PTDataLoader(val_ds, batch_size=args.batchsize, shuffle=False)
        test_loader = PTDataLoader(test_ds, batch_size=args.batchsize, shuffle=False)

        model = torch.nn.Linear(args.hidden, 1).to(device)
        best_model = copy.deepcopy(model.state_dict())
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=50 // args.eval_every,
                                                         min_lr=1.e-5)

        trainer = LinearTrainer(args.losstype)

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss = trainer.train(train_loader, model, optimizer)
            stats_dict = {'train_loss': train_loss,
                          'lr': scheduler.optimizer.param_groups[0]["lr"]}
            if epoch % args.eval_every == 0:
                val_loss = trainer.eval(val_loader, model)

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

                if trainer.patience > (args.patience // args.eval_every + 1):
                    break

                stats_dict['val_loss'] = val_loss

            pbar.set_postfix(stats_dict)
            wandb.log(stats_dict)

        best_val_losses.append(trainer.best_val_loss)

        model.load_state_dict(best_model)
        test_loss = trainer.eval(test_loader, model)
        test_losses.append(test_loss)
        wandb.log({'test_loss': test_loss})

        wandb.log({
            'best_val_loss': np.mean(best_val_losses),
            'test_loss_mean': np.mean(test_losses),
            'test_loss_std': np.std(test_losses),
        })


if __name__ == '__main__':
    main()
