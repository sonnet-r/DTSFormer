import os

import numpy as np
import torch


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == "constant":
        lr = args.learning_rate
    elif args.lradj == "type1":
        lr = args.learning_rate * (0.5 ** (epoch - 1))
    elif args.lradj == "type2":
        schedule = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
        if epoch not in schedule:
            return
        lr = schedule[epoch]
    elif args.lradj in {"3", "4", "5", "6"}:
        decay_epoch = {"3": 10, "4": 15, "5": 25, "6": 5}[args.lradj]
        lr = args.learning_rate if epoch < decay_epoch else args.learning_rate * 0.1
    elif args.lradj == "TST":
        lr = scheduler.get_last_lr()[0]
    else:
        raise ValueError(f"Unknown learning-rate schedule: {args.lradj}")

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = lr * parameter_group.get("lr_scale", 1.0)
    if printout:
        print(f"Learning rate: {lr:g}")


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model, path):
        if not np.isfinite(val_loss):
            print(
                f"Validation loss is non-finite ({val_loss}); "
                "stopping without overwriting the best checkpoint"
            )
            self.early_stop = True
            return

        score = -val_loss
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self._save_checkpoint(val_loss, model, path)
            self.counter = 0
            return

        self.counter += 1
        print(f"Early stopping: {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True

    def _save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased "
                f"({self.val_loss_min:.6f} -> {val_loss:.6f}); saving checkpoint"
            )
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        self.val_loss_min = val_loss
