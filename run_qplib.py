import os
from collections import defaultdict
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
from models.hetero_gnn import GNN
from trainers.supervised_trainer import PlainGNNTrainer
from trainers.training_loops import supervised_train_loops
from transforms.gcn_norm import GCNNorm


@hydra.main(version_base=None, config_path='./config', config_name="run")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

    transform = GCNNorm() if 'gcn' in args.backbone.conv else None

    train_subset = LPDataset(args.exp.datapath, 'train', transform=transform)
    if args.exp.debug:
        train_subset = train_subset[:2]
    offset = train_subset.data.obj_solution.min().item() - 1.e-3  # To avoid log(0)
    train_subset.data.obj_solution = torch.log10(train_subset.data.obj_solution - offset)

    train_loader = DataLoader(train_subset,
                              batch_size=1,
                              shuffle=True,
                              collate_fn=collate_fn_lp_base,
                              pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_dict = defaultdict(list)
    train_losses = []

    if args.finetune.modelpath is not None:
        model_dicts = os.listdir(args.finetune.modelpath)
        model_dicts = [m for m in model_dicts if m.startswith('pretrain') and m.endswith('.pt')]
        runs = len(model_dicts)
    else:
        model_dicts = None
        runs = args.exp.runs

    for run in range(runs):
        model = GNN(is_qp=is_qp(train_subset[0]),
                    conv=args.backbone.conv,
                    hid_dim=args.backbone.hidden,
                    num_encode_layers=args.backbone.num_encode_layers,
                    num_conv_layers=args.backbone.num_conv_layers,
                    num_pred_layers=args.finetune.num_pred_layers,
                    num_mlp_layers=args.backbone.num_mlp_layers,
                    backbone_pred_layers=args.backbone.num_pred_layers,
                    norm=args.backbone.norm).to(device)
        if model_dicts is not None:
            state_dict = torch.load(os.path.join(args.finetune.modelpath, model_dicts[run]), map_location=device)
            model.encoder.load_state_dict(state_dict)

        optimizer = optim.Adam(model.parameters(), lr=args.finetune.lr, weight_decay=args.finetune.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=100,
                                                         min_lr=1.e-5)

        trainer = PlainGNNTrainer()

        best_model = supervised_train_loops(args.finetune.epoch, args.finetune.patience, args.exp.ckpt,
                                            run, 0, log_folder_name,
                                            trainer, train_loader, model, optimizer, scheduler)

        train_losses.append(trainer.best_objgap)
        model.load_state_dict(best_model)
        model.eval()

        with torch.no_grad():
            for instance in train_loader:
                instance = instance.to(device)
                qpid = f'QPLIB_{instance.qpid.item()}'
                predict_obj = model(instance).item()
                predict_obj = 10 ** predict_obj + offset
                eval_dict[qpid].append(predict_obj)

    summary_dict = {'train_loss_mean': np.mean(train_losses),
                    'train_loss_std': np.std(train_losses)}
    for instance in train_loader:
        qpid = f'QPLIB_{instance.qpid.item()}'
        summary_dict[f'{qpid}_gt'] = 10 ** instance.obj_solution.item() + offset
        summary_dict[f'{qpid}_mean'] = np.mean(eval_dict[qpid]).item()
        summary_dict[f'{qpid}_std'] = np.std(eval_dict[qpid]).item()

    wandb.log(summary_dict)


if __name__ == '__main__':
    main()
