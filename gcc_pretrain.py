import hydra
import torch
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose

from data.collate_func import collate_pos_pair
from data.dataset import LPDataset
from utils.experiment import save_run_config, setup_wandb
from models.infonce_pretrain_gnn import PretrainGNN
from trainers.ntxent_pretrainer import NTXentPretrainer
from trainers.training_loops import pretraining_loops
from transforms.gcn_norm import GCNNormDumb
from transforms.rw_subgraph import RWSubgraph
from transforms.wrapper import ComboAugmentWrapper


def pretrain(args: DictConfig, log_folder_name: str = None, run_id: int = 0):
    # the only augmentation
    aug_list = [RWSubgraph(args.pretrain.method.RWSubgraph.walk_length)]
    transform = [ComboAugmentWrapper(aug_list)]

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


@hydra.main(version_base=None, config_path='./config', config_name="gcc")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

    for run in range(args.exp.runs):
        pretrain(args, log_folder_name, run)


if __name__ == '__main__':
    main()
