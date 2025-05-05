import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose

from data.collate_func import collate_pos_pair
from data.dataset import LPDataset
from finetune import finetune
from models.infonce_pretrain_gnn import PretrainGNN
from trainers.ntxent_pretrainer import NTXentPretrainer
from trainers.training_loops import pretraining_loops
from transforms.gcn_norm import GCNNormDumb
from transforms.graph_cl import ComboGraphCL
from utils.evaluation import is_qp
from utils.experiment import save_run_config, setup_wandb

torch.multiprocessing.set_sharing_strategy('file_system')


def pretrain(args: DictConfig, log_folder_name: str = None, run_id: int = 0):
    aug_dict = {aug_class: kwargs.strength for aug_class, kwargs in args.pretrain.method.items() if kwargs.strength > 0.}
    transform = [ComboGraphCL(aug_dict)]

    # Don't use GCNnorm during pretraining! It makes the pretraining converge too fast!
    if 'gcn' in args.backbone.conv:
        transform.append(GCNNormDumb())
    transform = Compose(transform)
    train_set = LPDataset(args.exp.datapath, 'train', transform=transform)
    if args.exp.debug:
        train_set = train_set[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collate_fn = collate_pos_pair
    train_loader = DataLoader(train_set,
                              batch_size=args.pretrain.batchsize,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=4,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=4)

    model = PretrainGNN(
        is_qp=is_qp(train_set[0]),
        conv=args.backbone.conv,
        hid_dim=args.backbone.hidden,
        num_encode_layers=args.backbone.num_encode_layers,
        num_conv_layers=args.backbone.num_conv_layers,
        num_pred_layers=args.pretrain.num_pred_layers,
        num_mlp_layers=args.backbone.num_mlp_layers,
        backbone_pred_layers=args.backbone.num_pred_layers,
        norm=args.backbone.norm).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.pretrain.lr, weight_decay=args.pretrain.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=50,
                                                     min_lr=1.e-5)

    trainer = NTXentPretrainer(args.pretrain.temperature)

    best_model = pretraining_loops(args.pretrain.epoch, args.pretrain.patience, args.exp.ckpt,
                                   run_id, log_folder_name,
                                   trainer, train_loader, device, model, optimizer, scheduler)
    return best_model


@hydra.main(version_base=None, config_path='./config', config_name="graphcl")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

    best_val_objgaps = []
    test_objgaps = []

    for run in range(args.exp.runs):
        pretrained_model = pretrain(args, log_folder_name, run)
        val_obj, test_obj = finetune(args, log_folder_name, run, pretrained_model)

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
