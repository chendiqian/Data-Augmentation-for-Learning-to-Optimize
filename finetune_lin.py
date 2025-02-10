import os

import hydra
import copy
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from data.dataset import LPDataset
from data.collate_func import collate_fn_lp_base
from data.transforms import GCNNorm
from mol_models.encoder import Encoder
from trainer import LinearTrainer
from data.utils import save_run_config


@hydra.main(version_base=None, config_path='./config', config_name="finetune")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)

    wandb.init(project=args.wandb.project,
               name=args.wandb.name if args.wandb.name else None,
               mode="online" if args.wandb.enable else "disabled",
               config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
               entity="chendiqian")  # use your own entity

    train_set = LPDataset(args.datapath, 'train', transform=GCNNorm() if 'gcn' in args.conv else None)
    if args.train_frac < 1:
        train_set = train_set[:int(len(train_set) * args.train_frac)]
    valid_set = LPDataset(args.datapath, 'valid', transform=GCNNorm() if 'gcn' in args.conv else None)
    test_set = LPDataset(args.datapath, 'test', transform=GCNNorm() if 'gcn' in args.conv else None)
    if args.debug:
        train_set = train_set[:20]
        valid_set = valid_set[:20]
        test_set = test_set[:20]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(train_set,
                              batch_size=args.batchsize,
                              shuffle=False,
                              collate_fn=collate_fn_lp_base)
    val_loader = DataLoader(valid_set,
                            batch_size=args.val_batchsize,
                            shuffle=False,
                            collate_fn=collate_fn_lp_base)
    test_loader = DataLoader(test_set,
                             batch_size=args.val_batchsize,
                             shuffle=False,
                             collate_fn=collate_fn_lp_base)

    best_val_objgaps = []
    test_objgaps = []

    model = Encoder(conv=args.conv,
                    hid_dim=args.hidden,
                    num_conv_layers=args.num_conv_layers,
                    num_pred_layers=args.num_pred_layers,
                    num_mlp_layers=args.num_mlp_layers,
                    norm=args.norm).to(device)
    model.load_state_dict(torch.load(args.modelpath, map_location=device))

    def get_feat_label(loader):
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                obj_pred = model(data)
                label = data.obj_solution
                features.append(obj_pred)
                labels.append(label)
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        return TensorDataset(features, labels)

    train_ds = get_feat_label(train_loader)
    val_ds = get_feat_label(val_loader)
    test_ds = get_feat_label(test_loader)

    train_loader = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batchsize, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.val_batchsize, shuffle=False)

    for run in range(args.runs):
        model = torch.nn.Linear(args.hidden, 1).to(device)
        best_model = copy.deepcopy(model.state_dict())
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=50 // args.eval_every,
                                                         min_lr=1.e-5)

        trainer = LinearTrainer(args.losstype)

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss = trainer.train(train_loader, model, optimizer)
            stats_dict = {'train_loss': train_loss,
                          'lr': scheduler.optimizer.param_groups[0]["lr"]}
            if epoch % args.eval_every == 0:
                val_obj_gap = trainer.eval(val_loader, model)

                if scheduler is not None:
                    scheduler.step(val_obj_gap)

                if trainer.best_objgap > val_obj_gap:
                    trainer.patience = 0
                    trainer.best_objgap = val_obj_gap
                    best_model = copy.deepcopy(model.state_dict())
                    if args.ckpt:
                        torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
                else:
                    trainer.patience += 1

                if trainer.patience > (args.patience // args.eval_every + 1):
                    break

                stats_dict['val_obj_gap'] = val_obj_gap

            pbar.set_postfix(stats_dict)
            wandb.log(stats_dict)
        best_val_objgaps.append(trainer.best_objgap)

        model.load_state_dict(best_model)
        test_obj_gap = trainer.eval(test_loader, model)
        test_objgaps.append(test_obj_gap)
        wandb.log({'test_obj_gap': test_obj_gap})

    print(best_val_objgaps, test_obj_gap)
    wandb.log({
        'best_val_obj_gap': np.mean(best_val_objgaps),
        'test_obj_gap_mean': np.mean(test_objgaps),
        'test_obj_gap_std': np.std(test_objgaps),
    })


if __name__ == '__main__':
    main()
