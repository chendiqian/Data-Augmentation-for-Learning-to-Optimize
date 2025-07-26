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
from models.hetero_gnn import GNN
from trainers.supervised_trainer import QPLIBTrainer
from trainers.training_loops import supervised_train_eval_loops


@hydra.main(version_base=None, config_path='./config', config_name="run")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

    train_subset = LPDataset(args.exp.datapath, 'train')
    valid_subset = LPDataset(args.exp.datapath, 'valid')
    test_subset = LPDataset(args.exp.datapath, 'test')

    if args.exp.debug:
        # small ones
        train_subset = train_subset[3:5]
        valid_subset = valid_subset[3:5]
        test_subset = test_subset[3:5]

    offset = train_subset.data.obj_solution.min().item() - 1.e-3  # To avoid log(0)
    train_subset.data.obj_solution = torch.log10(train_subset.data.obj_solution - offset)
    offset = valid_subset.data.obj_solution.min().item() - 1.e-3
    valid_subset.data.obj_solution = torch.log10(valid_subset.data.obj_solution - offset)
    offset = test_subset.data.obj_solution.min().item() - 1.e-3
    test_subset.data.obj_solution = torch.log10(test_subset.data.obj_solution - offset)

    train_loader = DataLoader(train_subset,
                              batch_size=1,
                              shuffle=False,
                              collate_fn=collate_fn_lp_base,
                              pin_memory=True)
    valid_loader = DataLoader(valid_subset,
                              batch_size=1,
                              shuffle=False,
                              collate_fn=collate_fn_lp_base,
                              pin_memory=True)
    test_loader = DataLoader(test_subset,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=collate_fn_lp_base,
                             pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_dict = defaultdict(list)

    for run in range(args.exp.runs):
        model = GNN(is_qp=True,
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

        trainer = QPLIBTrainer()

        best_model = supervised_train_eval_loops(
            args.finetune.epoch, args.finetune.patience, args.exp.ckpt,
            run, 0, log_folder_name,
            trainer, train_loader, valid_loader, device, model, optimizer, scheduler)

        model.load_state_dict(best_model)
        model.eval()

        with torch.no_grad():
            for instance in test_loader:
                instance = instance.to(device)
                qpid = f'QPLIB_{instance.qpid.item()}'
                predict_obj = model(instance).item()
                predict_obj = 10 ** predict_obj + offset
                eval_dict[qpid].append(predict_obj)

    summary_dict = {}
    for instance in test_loader:
        qpid = f'QPLIB_{instance.qpid.item()}'
        summary_dict[f'{qpid}_gt'] = 10 ** instance.obj_solution.item() + offset
        summary_dict[f'{qpid}_mean'] = np.mean(eval_dict[qpid]).item()
        summary_dict[f'{qpid}_std'] = np.std(eval_dict[qpid]).item()

    wandb.log(summary_dict)


if __name__ == '__main__':
    main()
