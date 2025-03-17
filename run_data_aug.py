import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose

import transforms
from data.collate_func import collate_fn_lp_base
from data.dataset import LPDataset
from utils.experiment import save_run_config, setup_wandb
from models.hetero_gnn import GNN
from trainers.supervised_trainer import PlainGNNTrainer
from trainers.training_loops import supervised_train_eval_loops
from transforms.gcn_norm import GCNNorm
from transforms.wrapper import SingleAugmentWrapper


@hydra.main(version_base=None, config_path='./config', config_name="run_data_aug")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

    aug_list = [getattr(transforms, aug_class)(**kwargs)
                for aug_class, kwargs in args.data_aug.method.items() if kwargs.strength > 0.]
    transform = [SingleAugmentWrapper(aug_list)]

    if 'gcn' in args.backbone.conv:
        extra_transform = GCNNorm()
        transform.append(extra_transform)
    else:
        extra_transform = None
    transform = Compose(transform)

    train_set = LPDataset(args.exp.datapath, 'train', transform=transform)
    if args.finetune.train_frac < 1:
        train_set = train_set[:int(len(train_set) * args.finetune.train_frac)]
    valid_set = LPDataset(args.exp.datapath, 'valid', transform=extra_transform)
    test_set = LPDataset(args.exp.datapath, 'test', transform=extra_transform)
    if args.exp.debug:
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

    best_val_objgaps = []
    test_objgaps = []

    for run in range(args.exp.runs):
        model = GNN(conv=args.backbone.conv,
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
                                                 run, 0, log_folder_name,
                                                 trainer, train_loader, val_loader, device, model, optimizer, scheduler)

        best_val_objgaps.append(trainer.best_objgap)

        model.load_state_dict(best_model)
        test_obj_gap = trainer.eval(test_loader, model)
        test_objgaps.append(test_obj_gap)
        wandb.log({'test_obj_gap': test_obj_gap})

    wandb.log({
        'best_val_obj_gap': np.mean(best_val_objgaps),
        'test_obj_gap_mean': np.mean(test_objgaps),
        'test_obj_gap_std': np.std(test_objgaps)
    })


if __name__ == '__main__':
    main()
