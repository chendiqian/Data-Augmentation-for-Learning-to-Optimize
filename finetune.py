import os

import hydra
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import MLP
import wandb
from omegaconf import DictConfig

from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_base
from transforms.gcn_norm import GCNNorm
from models.hetero_gnn import GNN
from models.backbone import Backbone
from trainers.supervised_trainer import PlainGNNTrainer, LinearTrainer
from trainers.training_loops import supervised_train_eval_loops
from utils.experiment import save_run_config, setup_wandb


def get_feat_label(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            obj_pred, *_ = model(data)
            label = data.obj_solution
            features.append(obj_pred)
            labels.append(label)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return TensorDataset(features, labels)


def finetune(args: DictConfig, log_folder_name: str = None, run_id: int = 0, pretrained_state_dict=None):
    transform = GCNNorm() if 'gcn' in args.backbone.conv else None

    max_folds = 1. / args.finetune.train_frac
    # make sure the fraction can be divided
    assert max_folds == int(max_folds)
    max_folds = min(int(max_folds), args.finetune.folds)

    train_set = LPDataset(args.exp.datapath, 'train', transform=transform)
    valid_set = LPDataset(args.exp.datapath, 'valid', transform=transform)
    test_set = LPDataset(args.exp.datapath, 'test', transform=transform)
    if args.exp.debug:
        valid_set = valid_set[:20]
        test_set = test_set[:20]

    is_qp = ('vals', 'to', 'vals') in train_set[0].edge_index_dict

    val_loader = DataLoader(valid_set,
                            batch_size=args.finetune.batchsize,
                            shuffle=False,
                            collate_fn=collate_fn_lp_base,
                            pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=args.finetune.batchsize,
                             shuffle=False,
                             collate_fn=collate_fn_lp_base,
                             pin_memory=True)

    ndata_per_fold = int(len(train_set) * args.finetune.train_frac)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_val_obj_gap_across_folds = []
    test_obj_gap_across_folds = []

    for _fold in range(max_folds):
        train_subset = train_set[_fold * ndata_per_fold: (_fold + 1) * ndata_per_fold]
        if args.exp.debug:
            train_subset = train_subset[:20]

        train_loader = DataLoader(train_subset,
                                  batch_size=args.finetune.batchsize,
                                  shuffle=True,
                                  collate_fn=collate_fn_lp_base,
                                  pin_memory=True)

        if args.finetune.whole:
            # finetune the whole model
            model = GNN(is_qp=is_qp,
                        conv=args.backbone.conv,
                        hid_dim=args.backbone.hidden,
                        num_encode_layers=args.backbone.num_encode_layers,
                        num_conv_layers=args.backbone.num_conv_layers,
                        num_pred_layers=args.finetune.num_pred_layers,
                        num_mlp_layers=args.backbone.num_mlp_layers,
                        backbone_pred_layers=args.backbone.num_pred_layers,
                        norm=args.backbone.norm).to(device)
            model.encoder.load_state_dict(pretrained_state_dict)
            trainer = PlainGNNTrainer()
        else:
            model = Backbone(
                is_qp=is_qp,
                conv=args.backbone.conv,
                hid_dim=args.backbone.hidden,
                num_encode_layers=args.backbone.num_encode_layers,
                num_conv_layers=args.backbone.num_conv_layers,
                num_mlp_layers=args.backbone.num_mlp_layers,
                backbone_pred_layers=args.backbone.num_pred_layers,
                norm=args.backbone.norm).to(device)
            model.load_state_dict(pretrained_state_dict)

            train_ds = get_feat_label(model, train_loader, device)
            val_ds = get_feat_label(model, val_loader, device)
            test_ds = get_feat_label(model, test_loader, device)

            train_loader = DataLoader(train_ds, batch_size=args.finetune.batchsize, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=args.finetune.batchsize, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=args.finetune.batchsize, shuffle=False)

            model = MLP([args.backbone.hidden] * args.finetune.num_pred_layers + [1], norm=None).to(device)
            trainer = LinearTrainer()

        optimizer = optim.Adam(model.parameters(), lr=args.finetune.lr, weight_decay=args.finetune.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=100,
                                                         min_lr=1.e-5)

        best_model = supervised_train_eval_loops(args.finetune.epoch, args.finetune.patience, args.exp.ckpt,
                                                 run_id, _fold, log_folder_name,
                                                 trainer, train_loader, val_loader, device, model, optimizer, scheduler)

        model.load_state_dict(best_model)
        test_obj_gap = trainer.eval(test_loader, model)
        best_val_obj_gap_across_folds.append(trainer.best_objgap)
        test_obj_gap_across_folds.append(test_obj_gap)

    return np.mean(best_val_obj_gap_across_folds), np.mean(test_obj_gap_across_folds)


@hydra.main(version_base=None, config_path='./config', config_name="pre_fine")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_val_objgaps = []
    test_objgaps = []

    assert args.finetune.modelpath is not None

    model_dicts = os.listdir(args.finetune.modelpath)
    model_dicts = [m for m in model_dicts if m.startswith('pretrain') and m.endswith('.pt')]

    for run, model_dict in enumerate(model_dicts):
        state_dict = torch.load(os.path.join(args.finetune.modelpath, model_dict), map_location=device)
        val_obj, test_obj = finetune(args, log_folder_name, run, state_dict)

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
