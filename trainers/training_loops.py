import copy
import os

import torch
import wandb
from tqdm import tqdm

from data.prefetch_generator import BackgroundGenerator
from utils.models import average_weights


def supervised_train_eval_loops(epochs, patience,
                                ckpt, run_id, fold_id, log_folder_name,
                                trainer, train_loader, val_loader, device, model, optimizer, scheduler):
    pbar = tqdm(range(epochs))
    best_model = copy.deepcopy(model.state_dict())
    for epoch in pbar:
        train_loss = trainer.train(train_loader, model, optimizer)
        val_obj_gap = trainer.eval(val_loader, model)

        if scheduler is not None:
            scheduler.step(val_obj_gap)

        if trainer.best_objgap > val_obj_gap:
            trainer.patience = 0
            trainer.best_objgap = val_obj_gap
            best_model = copy.deepcopy(model.state_dict())
            if ckpt:
                torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run_id}_{fold_id}.pt'))
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


def pretraining_loops(epochs, patience,
                      ckpt, run_id, log_folder_name,
                      trainer, train_loader, device, model, optimizer, scheduler):
    pbar = tqdm(range(epochs))
    best_model = copy.deepcopy(model.encoder.state_dict())
    for epoch in pbar:
        train_loss, train_acc = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)

        if scheduler is not None:
            scheduler.step(train_loss)

        if trainer.best_loss > train_loss:
            trainer.patience = 0
            trainer.best_loss = train_loss
            best_model = copy.deepcopy(model.encoder.state_dict())
            if ckpt:
                torch.save(model.encoder.state_dict(), os.path.join(log_folder_name, f'pretrain_best_model{run_id}.pt'))
        else:
            trainer.patience += 1

        if ckpt and epoch % 100 == 99:
            torch.save(model.encoder.state_dict(), os.path.join(log_folder_name, f'pretrain_model_epoch{epoch}.pt'))

        if trainer.patience > patience:
            break

        stats_dict = {'pretrain_loss': train_loss,
                      'pretrain_acc': train_acc,
                      'pretrain_lr': scheduler.optimizer.param_groups[0]["lr"]}

        pbar.set_postfix(stats_dict)
        wandb.log(stats_dict)
    return best_model


def siamese_pretraining_loops(epochs, patience,
                              ckpt, run_id, log_folder_name,
                              trainer, train_loader, device, model, optimizer, scheduler):
    """
    This is for some pretraining, where at the end the two pretraining nets are merged to give a final model
    e.g. IGSD the distillation, teacher and student nets
    MVGRL, where the GNN encoder is not shared but the MLP is shared

    Args:
        epochs:
        patience:
        ckpt:
        run_id:
        log_folder_name:
        trainer:
        train_loader:
        device:
        model:
        optimizer:
        scheduler:

    Returns:

    """
    pbar = tqdm(range(epochs))
    # the best is the mean of 2 nets
    best_model = average_weights([model.encoder1.state_dict(), model.encoder2.state_dict()])
    for epoch in pbar:
        train_loss, train_acc = trainer.train(BackgroundGenerator(train_loader, device, 4), model, optimizer)

        if scheduler is not None:
            scheduler.step(train_loss)

        if trainer.best_loss > train_loss:
            trainer.patience = 0
            trainer.best_loss = train_loss
            best_model = average_weights([model.encoder1.state_dict(), model.encoder2.state_dict()])
            if ckpt:
                torch.save(best_model, os.path.join(log_folder_name, f'pretrain_best_model{run_id}.pt'))
        else:
            trainer.patience += 1

        if trainer.patience > patience:
            break

        stats_dict = {'pretrain_loss': train_loss,
                      'pretrain_acc': train_acc,
                      'pretrain_lr': scheduler.optimizer.param_groups[0]["lr"]}

        pbar.set_postfix(stats_dict)
        wandb.log(stats_dict)
    return best_model
