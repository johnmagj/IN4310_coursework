import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import average_precision_score

from dataclasses import dataclass
from typing import Callable, Dict, Optional

@dataclass
class EpochStats:
    loss: float
    acc: float
    mAP: float
    AP_per_class: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        augment_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.augment_fn = augment_fn

        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_mAP": [],
            "val_AP_per_class": [],
        }

    @staticmethod
    def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()

    def train_one_epoch(self, loader: DataLoader) -> EpochStats:
        self.model.train()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Apply augmentation policy only during training (optional)
            if self.augment_fn is not None:
                images = self.augment_fn(images)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += self._accuracy(logits, targets)
            n_batches += 1

        return EpochStats(loss=total_loss / n_batches, acc=total_acc / n_batches, mAP=0.0, AP_per_class=0.0)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EpochStats:
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        store_logits = []
        store_targets = []

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            total_loss += loss.item()
            total_acc += self._accuracy(logits, targets)
            n_batches += 1

            store_logits.append(logits)
            store_targets.append(targets)

        # logits have shape [batch size, number of classes], 
        # first row is the raw output from the model for the first individual sample of the batch.
        # Turn list of tensors (one for each batch) into one tensor-typed list
        all_logits = torch.concatenate(store_logits)
        all_targets = torch.concatenate(store_targets)

        # Move to cpu
        all_logits = all_logits.cpu()
        all_targets = all_targets.cpu()

        # Turn raw model output into probabilities with softmax, dim=1 so we get sofmax per indv. sample over all classes
        prob_scores = torch.softmax(all_logits, dim=1).numpy()
        
        # Turn target values (class number) into one-hot encoded tensor 
        n_classes = all_logits.shape[1]
        one_hot_targets = torch.nn.functional.one_hot(all_targets, num_classes=n_classes).numpy()

        mean_avg_precision = average_precision_score(y_score=prob_scores, y_true=one_hot_targets)
        avg_precision_per_class = average_precision_score(y_score=prob_scores, y_true=one_hot_targets, average=None)

        return EpochStats(loss=total_loss/n_batches, 
                          acc=total_acc/n_batches, 
                          mAP=mean_avg_precision, 
                          AP_per_class=avg_precision_per_class)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10):
        for epoch in range(1, epochs + 1):
            train_stats = self.train_one_epoch(train_loader)
            val_stats = self.evaluate(val_loader)

            self.history["train_loss"].append(train_stats.loss)
            self.history["train_acc"].append(train_stats.acc)
            self.history["val_loss"].append(val_stats.loss)
            self.history["val_acc"].append(val_stats.acc)
            self.history["val_mAP"].append(val_stats.mAP)
            self.history["val_AP_per_class"].append(val_stats.AP_per_class)

            print(
                f"Epoch {epoch:02d} | "
                f"train loss: {train_stats.loss:.4f}, acc: {train_stats.acc:.3f} | "
                f"val loss: {val_stats.loss:.4f}, acc: {val_stats.acc:.3f}, mean average precision (mAP): {val_stats.mAP:.3f}, AP per class: {val_stats.AP_per_class}"
            )
