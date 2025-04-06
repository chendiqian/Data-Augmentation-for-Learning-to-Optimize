import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader

from data.collate_func import collate_fn_lp_base
from data.dataset import LPDataset
from utils.experiment import save_run_config, setup_wandb
from utils.evaluation import is_qp
from utils.models import count_parameters
from models.hetero_gnn import GNN
from trainers.supervised_trainer import PlainGNNTrainer
from trainers.training_loops import supervised_train_eval_loops
from transforms.gcn_norm import GCNNorm


@hydra.main(version_base=None, config_path='./config', config_name="run")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

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
    best_val_objgaps = []
    test_objgaps = []

    for run in range(args.exp.runs):
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

            model = GNN(is_qp=is_qp(train_set[0]),
                        conv=args.backbone.conv,
                        hid_dim=args.backbone.hidden,
                        num_encode_layers=args.backbone.num_encode_layers,
                        num_conv_layers=args.backbone.num_conv_layers,
                        num_pred_layers=args.finetune.num_pred_layers,
                        num_mlp_layers=args.backbone.num_mlp_layers,
                        backbone_pred_layers=args.backbone.num_pred_layers,
                        norm=args.backbone.norm).to(device)

            optimizer = optim.Adam(model.parameters(), lr=args.finetune.lr, weight_decay=args.finetune.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             factor=0.5,
                                                             patience=100,
                                                             min_lr=1.e-5)

            trainer = PlainGNNTrainer()

            best_model = supervised_train_eval_loops(args.finetune.epoch, args.finetune.patience, args.exp.ckpt,
                                                     run, _fold, log_folder_name,
                                                     trainer, train_loader, val_loader, device, model, optimizer, scheduler)
            model.load_state_dict(best_model)
            test_obj_gap = trainer.eval(test_loader, model)

            best_val_obj_gap_across_folds.append(trainer.best_objgap)
            test_obj_gap_across_folds.append(test_obj_gap)

        best_val_objgaps.append(np.mean(best_val_obj_gap_across_folds))
        test_objgaps.append(np.mean(test_obj_gap_across_folds))

    wandb.log({
        'num_params': count_parameters(model),
        'best_val_obj_gap': np.mean(best_val_objgaps),
        'test_obj_gap_mean': np.mean(test_objgaps),
        'test_obj_gap_std': np.std(test_objgaps)
    })


if __name__ == '__main__':
    main()
