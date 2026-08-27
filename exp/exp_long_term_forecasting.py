import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DTSFormer
from utils.metrics import EPSILON
from utils.tools import EarlyStopping, adjust_learning_rate


class ExpLongTermForecast(Exp_Basic):
    def _build_model(self) -> nn.Module:
        model = DTSFormer.Model(self.args).float()
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        print(f"Trainable parameters: {trainable:,}")
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _forward_batch(self, batch):
        batch_x, batch_y, _, _ = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        outputs = self.model(batch_x)
        feature_start = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, -self.args.pred_len :, feature_start:]
        targets = batch_y[:, -self.args.pred_len :, feature_start:]
        return outputs, targets

    def _evaluate_loss(self, loader, criterion):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                outputs, targets = self._forward_batch(batch)
                losses.append(criterion(outputs, targets).item())
        self.model.train()
        if not losses:
            raise RuntimeError("The evaluation loader produced no batches")
        return float(np.mean(losses))

    def train(self, setting):
        _, train_loader = self._get_data("train")
        _, validation_loader = self._get_data("val")
        checkpoint_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(checkpoint_dir, exist_ok=True)
        history_path = os.path.join(checkpoint_dir, "history.jsonl")

        optimizer = self._select_optimizer()
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
        )
        scheduler = None
        if self.args.lradj == "TST":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.args.learning_rate,
                steps_per_epoch=math.ceil(
                    len(train_loader) / self.args.accumulation_steps
                ),
                epochs=self.args.train_epochs,
                pct_start=self.args.pct_start,
            )

        scaler = None
        if self.args.use_amp:
            try:
                scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler()

        completed_epochs = 0
        with open(history_path, "w", encoding="utf-8") as history_file:
            for epoch in range(1, self.args.train_epochs + 1):
                if scheduler is None:
                    adjust_learning_rate(
                        optimizer,
                        None,
                        epoch,
                        self.args,
                        printout=epoch > 1,
                    )

                start = time.time()
                train_losses = []
                self.model.train()
                optimizer.zero_grad(set_to_none=True)
                for batch_index, batch in enumerate(train_loader, start=1):
                    if self.args.use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs, targets = self._forward_batch(batch)
                            loss = criterion(outputs, targets)
                        scaler.scale(loss / self.args.accumulation_steps).backward()
                    else:
                        outputs, targets = self._forward_batch(batch)
                        loss = criterion(outputs, targets)
                        (loss / self.args.accumulation_steps).backward()

                    train_losses.append(loss.item())
                    should_step = (
                        batch_index % self.args.accumulation_steps == 0
                        or batch_index == len(train_loader)
                    )
                    if should_step:
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if scheduler is not None:
                            scheduler.step()

                train_loss = float(np.mean(train_losses))
                validation_loss = self._evaluate_loss(validation_loader, criterion)
                completed_epochs = epoch
                record = {
                    "epoch": epoch,
                    "train_mse": train_loss,
                    "val_mse": validation_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                history_file.write(json.dumps(record) + "\n")
                history_file.flush()
                print(
                    f"Epoch {epoch}/{self.args.train_epochs} | "
                    f"train {train_loss:.6f} | val {validation_loss:.6f} | "
                    f"{time.time() - start:.1f}s"
                )
                early_stopping(validation_loss, self.model, checkpoint_dir)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        summary = {
            "completed_epochs": completed_epochs,
            "best_val_mse": float(early_stopping.val_loss_min),
        }
        with open(
            os.path.join(checkpoint_dir, "training_summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summary, file, indent=2)

        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return self.model

    def test(self, setting, load_checkpoint=False):
        _, test_loader = self._get_data("test")
        if load_checkpoint:
            checkpoint_path = os.path.join(
                self.args.checkpoints,
                setting,
                "checkpoint.pth",
            )
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )

        predictions = [] if self.args.save_predictions else None
        targets = [] if self.args.save_predictions else None
        element_count = 0
        sample_count = 0
        absolute_error = 0.0
        squared_error = 0.0
        absolute_percentage_error = 0.0
        squared_percentage_error = 0.0
        sum_prediction = None
        sum_target = None
        sum_prediction_square = None
        sum_target_square = None
        sum_cross_product = None

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                outputs, batch_targets = self._forward_batch(batch)
                prediction = outputs.cpu().numpy().astype(np.float64, copy=False)
                target = batch_targets.cpu().numpy().astype(np.float64, copy=False)
                difference = prediction - target
                denominator = np.maximum(np.abs(target), EPSILON)

                element_count += difference.size
                sample_count += difference.shape[0]
                absolute_error += np.abs(difference).sum(dtype=np.float64)
                squared_error += np.square(difference).sum(dtype=np.float64)
                relative_error = difference / denominator
                absolute_percentage_error += np.abs(relative_error).sum(dtype=np.float64)
                squared_percentage_error += np.square(relative_error).sum(dtype=np.float64)

                batch_sum_prediction = prediction.sum(axis=0, dtype=np.float64)
                batch_sum_target = target.sum(axis=0, dtype=np.float64)
                if sum_prediction is None:
                    sum_prediction = batch_sum_prediction
                    sum_target = batch_sum_target
                    sum_prediction_square = np.square(prediction).sum(
                        axis=0, dtype=np.float64
                    )
                    sum_target_square = np.square(target).sum(
                        axis=0, dtype=np.float64
                    )
                    sum_cross_product = (prediction * target).sum(
                        axis=0, dtype=np.float64
                    )
                else:
                    sum_prediction += batch_sum_prediction
                    sum_target += batch_sum_target
                    sum_prediction_square += np.square(prediction).sum(
                        axis=0, dtype=np.float64
                    )
                    sum_target_square += np.square(target).sum(
                        axis=0, dtype=np.float64
                    )
                    sum_cross_product += (prediction * target).sum(
                        axis=0, dtype=np.float64
                    )

                if predictions is not None:
                    predictions.append(prediction.astype(np.float32))
                    targets.append(target.astype(np.float32))

        if element_count == 0:
            raise RuntimeError("The test loader produced no batches")

        mae = absolute_error / element_count
        mse = squared_error / element_count
        rmse = math.sqrt(mse)
        mape = absolute_percentage_error / element_count
        mspe = squared_percentage_error / element_count
        target_centered_sum = (
            sum_target_square - np.square(sum_target) / sample_count
        ).sum(dtype=np.float64)
        rse = math.sqrt(squared_error) / max(
            math.sqrt(max(target_centered_sum, 0.0)), EPSILON
        )
        covariance = sum_cross_product - sum_prediction * sum_target / sample_count
        prediction_variance = (
            sum_prediction_square - np.square(sum_prediction) / sample_count
        )
        target_variance = sum_target_square - np.square(sum_target) / sample_count
        correlation = covariance / np.maximum(
            np.sqrt(np.maximum(prediction_variance * target_variance, 0.0)),
            EPSILON,
        )
        metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
            "rse": float(rse),
            "corr": float(np.mean(correlation)),
        }
        print(f"Test | MSE {mse:.6f} | MAE {mae:.6f} | RSE {rse:.6f}")

        result_dir = os.path.join(self.args.output_dir, setting)
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "metrics.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metrics, file, indent=2)
        if predictions is not None:
            np.save(
                os.path.join(result_dir, "pred.npy"),
                np.concatenate(predictions, axis=0),
            )
            np.save(
                os.path.join(result_dir, "true.npy"),
                np.concatenate(targets, axis=0),
            )
        return metrics
