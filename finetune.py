import os

import copy

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP
from torch.utils.data import TensorDataset, DataLoader as PTDataLoader
from tqdm import tqdm
import numpy as np
from torch_geometric.datasets import ZINC
from data.prefetch_generator import BackgroundGenerator
from data.transforms import GCNNorm
from data.utils import save_run_config
from mol_models.encoder import Encoder
from mol_models.gnn import BasicGNN
from trainer import LinearTrainer, PlainGNNTrainer


def finetune(args: DictConfig, log_folder_name: str = None, run_id: int = 0, pretrained_state_dict=None):
    transform = GCNNorm() if 'gcn' in args.conv else None
    finetune_train_set = ZINC('./datasets', subset=True, split='train', transform=transform)
    if args.finetune.train_frac < 1:
        finetune_train_set = finetune_train_set[:int(len(finetune_train_set) * args.finetune.train_frac)]
    finetune_valid_set = ZINC('./datasets', subset=True, split='val', transform=transform)
    finetune_test_set = ZINC('./datasets', subset=True, split='test', transform=transform)
    if args.debug:
        finetune_train_set = finetune_train_set[:20]
        finetune_valid_set = finetune_valid_set[:20]
        finetune_test_set = finetune_test_set[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    finetune_train_loader = DataLoader(finetune_train_set,
                                       batch_size=args.finetune.batchsize,
                                       shuffle=False)
    finetune_val_loader = DataLoader(finetune_valid_set,
                                     batch_size=args.finetune.batchsize,
                                     shuffle=False)
    finetune_test_loader = DataLoader(finetune_test_set,
                                      batch_size=args.finetune.batchsize,
                                      shuffle=False)

    if args.finetune.whole:
        train_loader = DataLoader(finetune_train_loader,
                                  batch_size=args.finetune.batchsize,
                                  shuffle=True)
        val_loader = DataLoader(finetune_val_loader,
                                batch_size=args.finetune.batchsize,
                                shuffle=False)
        test_loader = DataLoader(finetune_test_loader,
                                 batch_size=args.finetune.batchsize,
                                 shuffle=False)
        # finetune the whole model
        model = BasicGNN(conv=args.conv,
                         hid_dim=args.hidden,
                         num_conv_layers=args.num_conv_layers,
                         num_pred_layers=args.num_pred_layers,
                         num_mlp_layers=args.num_mlp_layers,
                         norm=args.norm).to(device)
        if pretrained_state_dict is not None:
            model.encoder.load_state_dict(pretrained_state_dict)

        best_model = copy.deepcopy(model.state_dict())
        # optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': 1.e-5},
        #                         {'params': model.predictor.parameters()}], lr=args.lr, weight_decay=args.weight_decay)

        trainer = PlainGNNTrainer(args.losstype)

    else:
        model = Encoder(conv=args.conv,
                        hid_dim=args.hidden,
                        num_conv_layers=args.num_conv_layers,
                        num_pred_layers=args.num_pred_layers,
                        num_mlp_layers=args.num_mlp_layers,
                        norm=args.norm).to(device)
        if pretrained_state_dict is not None:
            model.load_state_dict(pretrained_state_dict)

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

        train_loader = PTDataLoader(train_ds, batch_size=args.finetune.batchsize, shuffle=True)
        val_loader = PTDataLoader(val_ds, batch_size=args.finetune.batchsize, shuffle=False)
        test_loader = PTDataLoader(test_ds, batch_size=args.finetune.batchsize, shuffle=False)

        # todo: tune an mlp
        model = MLP([args.hidden] * args.finetune.num_mlp_layers + [1], norm=None).to(device)
        best_model = copy.deepcopy(model.state_dict())

        trainer = LinearTrainer(args.losstype)

    optimizer = optim.Adam(model.parameters(), lr=args.finetune.lr, weight_decay=args.finetune.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=50,
                                                     min_lr=1.e-5)

    pbar = tqdm(range(args.finetune.epoch))
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
                torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run_id}.pt'))
        else:
            trainer.patience += 1

        if trainer.patience > args.finetune.patience:
            break

        stats_dict = {'train_loss': train_loss,
                      'val_loss': val_loss,
                      'lr': scheduler.optimizer.param_groups[0]["lr"]}

        pbar.set_postfix(stats_dict)
        wandb.log(stats_dict)

    model.load_state_dict(best_model)
    test_loss = trainer.eval(test_loader, model)

    return best_model, trainer.best_val_loss, test_loss


@hydra.main(version_base=None, config_path='./config', config_name="dropnode")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_val_losses = []
    test_losses = []

    for run in range(args.runs):
        assert args.finetune.modelpath is not None
        state_dict = torch.load(args.finetune.modelpath, map_location=device)
        best_model, val_loss, test_loss = finetune(args, log_folder_name, run, state_dict)

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
