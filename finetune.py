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


def finetune(args: DictConfig, log_folder_name: str = None, run_id: int = 0, pretrained_state_dict=None):
    transform = GCNNorm() if 'gcn' in args.backbone.conv else None
    train_set = LPDataset(args.exp.datapath, 'train', transform=transform)
    if args.finetune.train_frac < 1:
        train_set = train_set[:int(len(train_set) * args.finetune.train_frac)]
    valid_set = LPDataset(args.exp.datapath, 'valid', transform=transform)
    test_set = LPDataset(args.exp.datapath, 'test', transform=transform)
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

    if args.finetune.whole:
        # finetune the whole model
        model = GNN(conv=args.backbone.conv,
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
            conv=args.backbone.conv,
            hid_dim=args.backbone.hidden,
            num_encode_layers=args.backbone.num_encode_layers,
            num_conv_layers=args.backbone.num_conv_layers,
            num_mlp_layers=args.backbone.num_mlp_layers,
            backbone_pred_layers=args.backbone.num_pred_layers,
            norm=args.backbone.norm).to(device)
        model.load_state_dict(pretrained_state_dict)
        def get_feat_label(loader):
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

        train_ds = get_feat_label(train_loader)
        val_ds = get_feat_label(val_loader)
        test_ds = get_feat_label(test_loader)

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
                                             run_id, log_folder_name,
                                             trainer, train_loader, val_loader, device, model, optimizer, scheduler)

    model.load_state_dict(best_model)
    test_obj_gap = trainer.eval(test_loader, model)
    return best_model, trainer.best_objgap, test_obj_gap


@hydra.main(version_base=None, config_path='./config', config_name="pre_fine")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_val_objgaps = []
    test_objgaps = []

    for run in range(args.exp.runs):
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
