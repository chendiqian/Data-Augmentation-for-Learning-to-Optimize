import copy
import os

import torch
import wandb
from tqdm import tqdm

from data.prefetch_generator import BackgroundGenerator


def supervised_train_eval_loops(epochs, patience,
                                ckpt, run_id, log_folder_name,
                                trainer, train_loader, val_loader, device, model, optimizer, scheduler):
    pbar = tqdm(range(epochs))
    best_model = copy.deepcopy(model.state_dict())
    for epoch in pbar:
        train_loss = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)
        val_obj_gap = trainer.eval(BackgroundGenerator(val_loader, device, 4), model)

        if scheduler is not None:
            scheduler.step(val_obj_gap)

        if trainer.best_objgap > val_obj_gap:
            trainer.patience = 0
            trainer.best_objgap = val_obj_gap
            best_model = copy.deepcopy(model.state_dict())
            if ckpt:
                torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run_id}.pt'))
        else:
            trainer.patience += 1

        if trainer.patience > patience:
            break

        stats_dict = {'train_loss': train_loss,
                      'val_obj_gap': val_obj_gap,
                      'lr': scheduler.optimizer.param_groups[0]["lr"]}

        pbar.set_postfix(stats_dict)
        wandb.log(stats_dict)
    return best_model


def pretraining_train_eval_loops(epochs, patience,
                                 ckpt, run_id, log_folder_name,
                                 trainer, train_loader, val_loader, device, model, optimizer, scheduler):
    pbar = tqdm(range(epochs))
    best_model = copy.deepcopy(model.encoder.state_dict())
    for epoch in pbar:
        train_loss, train_acc = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)
        val_loss, val_acc = trainer.eval(BackgroundGenerator(val_loader, device, 4), model)

        if scheduler is not None:
            scheduler.step(val_loss)

        if trainer.best_val_loss > val_loss:
            trainer.patience = 0
            trainer.best_val_loss = val_loss
            best_model = copy.deepcopy(model.encoder.state_dict())
            if ckpt:
                torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run_id}.pt'))
        else:
            trainer.patience += 1

        if trainer.patience > patience:
            break

        stats_dict = {'pretrain_train_loss': train_loss,
                      'pretrain_train_acc': train_acc,
                      'pretrain_val_loss': val_loss,
                      'pretrain_val_acc': val_acc,
                      'pretrain_lr': scheduler.optimizer.param_groups[0]["lr"]}

        pbar.set_postfix(stats_dict)
        wandb.log(stats_dict)
    return best_model
