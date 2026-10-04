import csv
import json
import math
import os

import numpy as np
import torch

from src.interfaces import TrainerInterface, ModelSaverInterface
from src.metrics import BinaryClassificationMetrics, YoudenThresholdSelector


class BestModelSaver(ModelSaverInterface):
    '''
    Persist the model whenever the monitored metric improves.
    mode='min' for losses, mode='max' for scores such as AUC.
    Optional metadata (epoch, threshold, metrics) is written to a JSON file next to the weights.
    '''
    def __init__(self, save_path: str, mode: str = 'min'):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.save_path = save_path
        self.mode = mode
        self.best_metric = math.inf if mode == 'min' else -math.inf

    @property
    def metadata_path(self) -> str:
        return os.path.splitext(self.save_path)[0] + '.json'

    def _improves(self, metric: float) -> bool:
        return metric < self.best_metric if self.mode == 'min' else metric > self.best_metric

    def save(self, model, metric: float, extra: dict | None = None) -> bool:
        if not self._improves(metric):
            return False

        print(f"  → New best! {self.best_metric:.4f} → {metric:.4f}. Saving...")
        self.best_metric = metric
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)

        if extra is not None:
            with open(self.metadata_path, 'w', encoding='utf-8') as metadata_file:
                json.dump(extra, metadata_file, indent=2)
        return True


class ModelTrainer(TrainerInterface):
    '''
    SRP: owns only the training and inference logic for a single epoch.
    DIP: criterion and optimizer are injected — nothing is hardcoded here.
    Mixed precision (bfloat16) is used on CUDA to cut memory and time on 3D volumes.
    '''
    def __init__(self, model, optimizer, criterion, device: torch.device, use_amp: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp)

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss = self.criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    def evaluate(self, loader) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_labels, all_probabilities = [], []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with self._autocast():
                    logits = self.model(images)
                total_loss += self.criterion(logits.float(), labels).item()

                all_probabilities.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        return total_loss / max(1, len(loader)), np.concatenate(all_labels), np.concatenate(all_probabilities)


class EarlyStopping:
    '''
    Stop training when the monitored metric has not improved for `patience` epochs.
    '''
    def __init__(self, patience: int = 15, mode: str = 'max', min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = -math.inf if mode == 'max' else math.inf
        self.epochs_without_improvement = 0

    def step(self, metric: float) -> bool:
        improved = (metric > self.best + self.min_delta) if self.mode == 'max' else (metric < self.best - self.min_delta)
        if improved:
            self.best = metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


class TrainingHistoryLogger:
    '''
    Append one row of metrics per epoch to a CSV file, for later plotting and comparison.
    '''
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if os.path.exists(csv_path):
            os.remove(csv_path)

    def log(self, row: dict) -> None:
        is_new = not os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as history_file:
            writer = csv.DictWriter(history_file, fieldnames=list(row))
            if is_new:
                writer.writeheader()
            writer.writerow(row)


class TrainingOrchestrator:
    '''
    SRP: coordinates the full epoch loop — unaware of internal training details.
    Model selection uses a validation metric (AUC by default) rather than the loss, and the decision
    threshold is tuned on validation at every epoch and stored with the best checkpoint.
    Scheduler, early stopping and history logging are optional collaborators.
    '''
    def __init__(
        self,
        trainer: TrainerInterface,
        saver: ModelSaverInterface,
        num_epochs: int = 30,
        monitor: str = 'auc',
        metrics: BinaryClassificationMetrics | None = None,
        threshold_selector: YoudenThresholdSelector | None = None,
        scheduler=None,
        early_stopping: EarlyStopping | None = None,
        history_logger: TrainingHistoryLogger | None = None,
    ):
        self.trainer = trainer
        self.saver = saver
        self.num_epochs = num_epochs
        self.monitor = monitor
        self.metrics = metrics or BinaryClassificationMetrics()
        self.threshold_selector = threshold_selector or YoudenThresholdSelector()
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.history_logger = history_logger

    def run(self, train_loader, val_loader) -> None:
        print(f"--- Starting Training & Validation Loop (monitoring val {self.monitor}) ---")

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.trainer.train_epoch(train_loader)
            val_loss, labels, probabilities = self.trainer.evaluate(val_loader)

            threshold = self.threshold_selector.select(labels, probabilities)
            results = self.metrics.compute(labels, probabilities, threshold)
            score = results[self.monitor]
            score = -math.inf if math.isnan(score) else score

            print(f"Epoch [{epoch}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"AUC: {results['auc']:.3f} | BalAcc: {results['balanced_accuracy']:.3f} | "
                  f"Sens: {results['sensitivity']:.3f} | Spec: {results['specificity']:.3f}", flush=True)

            record = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, **results}
            self.saver.save(self.trainer.model, score, extra=record)
            if self.history_logger is not None:
                self.history_logger.log({**record, 'lr': self.trainer.optimizer.param_groups[0]['lr']})

            if self.scheduler is not None:
                self.scheduler.step()
            if self.early_stopping is not None and self.early_stopping.step(score):
                print(f"Early stopping: no improvement in val {self.monitor} for {self.early_stopping.patience} epochs.")
                break

        print("Training complete.")
