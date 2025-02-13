import os

import hydra
import copy
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import MLP
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_base
from data.transforms import GCNNorm
from data.prefetch_generator import BackgroundGenerator
from models.hetero_gnn import TripartiteHeteroGNN
from models.hetero_backbone import TripartiteHeteroBackbone
from trainer import PlainGNNTrainer, LinearTrainer
from data.utils import save_run_config


def finetune(args: DictConfig, log_folder_name: str = None, run_id: int = 0, pretrained_state_dict=None):
    train_set = LPDataset(args.datapath, 'train', transform=GCNNorm() if 'gcn' in args.conv else None)
    if args.finetune.train_frac < 1:
        train_set = train_set[:int(len(train_set) * args.finetune.train_frac)]
    valid_set = LPDataset(args.datapath, 'valid', transform=GCNNorm() if 'gcn' in args.conv else None)
    test_set = LPDataset(args.datapath, 'test', transform=GCNNorm() if 'gcn' in args.conv else None)
    if args.debug:
        train_set = train_set[:20]
        valid_set = valid_set[:20]
        test_set = test_set[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(train_set,
                              batch_size=args.finetune.batchsize,
                              shuffle=True,
                              collate_fn=collate_fn_lp_base)
    val_loader = DataLoader(valid_set,
                            batch_size=args.finetune.batchsize,
                            shuffle=False,
                            collate_fn=collate_fn_lp_base)
    test_loader = DataLoader(test_set,
                             batch_size=args.finetune.batchsize,
                             shuffle=False,
                             collate_fn=collate_fn_lp_base)

    if args.finetune.whole:
        # finetune the whole model
        model = TripartiteHeteroGNN(conv=args.conv,
                                    hid_dim=args.hidden,
                                    num_encode_layers=args.num_encode_layers,
                                    num_conv_layers=args.num_conv_layers,
                                    num_pred_layers=args.num_pred_layers,
                                    num_mlp_layers=args.num_mlp_layers,
                                    backbone_pred_layers=args.backbone_pred_layers,
                                    norm=args.norm).to(device)
        model.encoder.load_state_dict(pretrained_state_dict)
        best_model = copy.deepcopy(model.state_dict())
        trainer = PlainGNNTrainer(args.losstype)
    else:
        model = TripartiteHeteroBackbone(
            conv=args.conv,
            hid_dim=args.hidden,
            num_encode_layers=args.num_encode_layers,
            num_conv_layers=args.num_conv_layers,
            num_mlp_layers=args.num_mlp_layers,
            backbone_pred_layers=args.backbone_pred_layers,
            norm=args.norm).to(device)
        model.load_state_dict(pretrained_state_dict)
        def get_feat_label(loader):
            model.eval()
            features = []
            labels = []
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    obj_pred = model(data)
                    label = data.obj_solution
                    features.append(obj_pred)
                    labels.append(label)
            features = torch.cat(features, dim=0)
            labels = torch.cat(labels, dim=0)
            return TensorDataset(features, labels)

        train_ds = get_feat_label(train_loader)
        val_ds = get_feat_label(val_loader)
        test_ds = get_feat_label(test_loader)

        train_loader = DataLoader(train_ds, batch_size=args.finetune.batchsize, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.finetune.batchsize, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.finetune.batchsize, shuffle=False)

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
        val_obj_gap = trainer.eval(BackgroundGenerator(val_loader, device, 4), model)

        if scheduler is not None:
            scheduler.step(val_obj_gap)

        if trainer.best_objgap > val_obj_gap:
            trainer.patience = 0
            trainer.best_objgap = val_obj_gap
            best_model = copy.deepcopy(model.state_dict())
            if args.ckpt:
                torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run_id}.pt'))
        else:
            trainer.patience += 1

        if trainer.patience > args.finetune.patience:
            break

        stats_dict = {'train_loss': train_loss,
                      'val_obj_gap': val_obj_gap,
                      'lr': scheduler.optimizer.param_groups[0]["lr"]}

        pbar.set_postfix(stats_dict)
        wandb.log(stats_dict)

    model.load_state_dict(best_model)
    test_obj_gap = trainer.eval(test_loader, model)
    return best_model, trainer.best_objgap, test_obj_gap


@hydra.main(version_base=None, config_path='./config', config_name="pre_fine")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_val_objgaps = []
    test_objgaps = []

    for run in range(args.runs):
        assert args.finetune.modelpath is not None
        state_dict = torch.load(args.finetune.modelpath, map_location=device)
        best_model, val_obj, test_obj = finetune(args, log_folder_name, run, state_dict)

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
