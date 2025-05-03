import copy
import os

import time
import torch
import wandb
from tqdm import tqdm

from utils.models import average_weights


def supervised_train_eval_loops(epochs, patience,
                                ckpt, run_id, fold_id, log_folder_name,
                                trainer, train_loader, val_loader, device, model, optimizer, scheduler):
    pbar = tqdm(range(epochs))
    times = []
    for epoch in pbar:
        t1 = time.time()
        train_loss = trainer.train(train_loader, model, optimizer)
        if epoch > 3:
            times.append(time.time() - t1)
    return times


def pretraining_loops(epochs, patience,
                      ckpt, run_id, log_folder_name,
                      trainer, train_loader, device, model, optimizer, scheduler):
    pbar = tqdm(range(epochs))
    times = []
    for epoch in pbar:
        t1 = time.time()
        _ = trainer.train(train_loader, model, optimizer)

        if epoch > 3:
            times.append(time.time() - t1)
    return times


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
        train_loss, train_acc = trainer.train(train_loader, model, optimizer)

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
