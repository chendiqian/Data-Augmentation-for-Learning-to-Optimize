import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

from data.collate_func import collate_pos_pair
from data.dataset import LPDataset
from data.prefetch_generator import BackgroundGenerator
from data.transforms import GCNNorm, RandomDropNode
from data.utils import save_run_config
from models.hetero_pretrain_gnn import BipartiteHeteroPretrainGNN, TripartiteHeteroPretrainGNN
from trainer import NTXentPretrainer


@hydra.main(version_base=None, config_path='./config', config_name="dropnode")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    # drop node first, then normalize degree
    transform = [RandomDropNode(args.drop_rate)]
    if 'gcn' in args.conv:
        transform.append(GCNNorm())
    transform = Compose(transform)
    train_set = LPDataset(args.datapath, 'train', transform=transform)
    valid_set = LPDataset(args.datapath, 'valid', transform=transform)
    if args.debug:
        train_set = train_set[:20]
        valid_set = valid_set[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(train_set,
                              batch_size=args.batchsize,
                              shuffle=True,
                              collate_fn=collate_pos_pair)
    val_loader = DataLoader(valid_set,
                            batch_size=args.val_batchsize,
                            shuffle=False,
                            collate_fn=collate_pos_pair)

    for run in range(args.runs):
        ModelClass = TripartiteHeteroPretrainGNN if args.tripartite else BipartiteHeteroPretrainGNN
        model = ModelClass(conv=args.conv,
                           head=args.gat.heads,
                           concat=args.gat.concat,
                           hid_dim=args.hidden,
                           num_encode_layers=args.num_encode_layers,
                           num_conv_layers=args.num_conv_layers,
                           num_pred_layers=args.num_pred_layers,
                           num_mlp_layers=args.num_mlp_layers,
                           norm=args.norm,
                           pooling=args.pooling).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='max',
                                                         factor=0.5,
                                                         patience=50 // args.eval_every,
                                                         min_lr=1.e-5)

        trainer = NTXentPretrainer(args.temperature)

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss, train_acc = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)
            stats_dict = {'train_loss': train_loss,
                          'train_top5acc': train_acc,
                          'lr': scheduler.optimizer.param_groups[0]["lr"]}
            if epoch % args.eval_every == 0:
                val_loss, val_acc = trainer.eval(BackgroundGenerator(val_loader, device, 4), model)

                if scheduler is not None:
                    scheduler.step(val_acc)

                if trainer.best_val_acc < val_acc:
                    trainer.patience = 0
                    trainer.best_val_acc = val_acc
                    if args.ckpt:
                        torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
                else:
                    trainer.patience += 1

                if trainer.patience > (args.patience // args.eval_every + 1):
                    break

                stats_dict['val_loss'] = val_loss
                stats_dict['val_top5acc'] = val_acc

            pbar.set_postfix(stats_dict)
            wandb.log(stats_dict)


if __name__ == '__main__':
    main()
