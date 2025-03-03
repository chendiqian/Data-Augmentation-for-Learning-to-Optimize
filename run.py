import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import DataLoader

from data.collate_func import collate_fn_lp_base
from data.dataset import LPDataset
from data.utils import save_run_config
from models.hetero_gnn import BipartiteHeteroGNN
from trainer import PlainGNNTrainer
from training_loops import supervised_train_eval_loops
from transforms.gcn_norm import GCNNorm


@hydra.main(version_base=None, config_path='./config', config_name="run")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    transform = GCNNorm() if 'gcn' in args.backbone.conv else None
    train_set = LPDataset(args.datapath, 'train', transform=transform)
    if args.train_frac < 1:
        train_set = train_set[:int(len(train_set) * args.train_frac)]
    valid_set = LPDataset(args.datapath, 'valid', transform=transform)
    test_set = LPDataset(args.datapath, 'test', transform=transform)
    if args.debug:
        train_set = train_set[:20]
        valid_set = valid_set[:20]
        test_set = test_set[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(train_set,
                              batch_size=args.batchsize,
                              shuffle=True,
                              collate_fn=collate_fn_lp_base)
    val_loader = DataLoader(valid_set,
                            batch_size=args.batchsize,
                            shuffle=False,
                            collate_fn=collate_fn_lp_base)
    test_loader = DataLoader(test_set,
                             batch_size=args.batchsize,
                             shuffle=False,
                             collate_fn=collate_fn_lp_base)

    best_val_objgaps = []
    test_objgaps = []

    for run in range(args.runs):
        model = BipartiteHeteroGNN(conv=args.backbone.conv,
                                   hid_dim=args.backbone.hidden,
                                   num_encode_layers=args.backbone.num_encode_layers,
                                   num_conv_layers=args.backbone.num_conv_layers,
                                   num_pred_layers=args.num_pred_layers,
                                   num_mlp_layers=args.backbone.num_mlp_layers,
                                   backbone_pred_layers=args.backbone.num_pred_layers,
                                   norm=args.backbone.norm).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=100,
                                                         min_lr=1.e-5)

        trainer = PlainGNNTrainer(args.losstype)

        best_model = supervised_train_eval_loops(args.epoch, args.patience, args.ckpt,
                                                 run,  log_folder_name,
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
