"""Model training and evaluation."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import config
from src.metrics import compute_all_metrics, print_metrics_summary


class ModelTrainer:
    """Trainer for spoof detection models."""

    def __init__(self, model, train_loader, val_loader, model_name="model", learning_rate=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.device = config.device
        self.learning_rate = learning_rate or config.learning_rate

        self.model.to(self.device)
        self._setup_training()
        self._init_history()

    def _setup_training(self):
        """Initialize loss function, optimizer, and scheduler."""
        class_weights = config.class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights: {config.class_weights.tolist()}")
        print(f"Learning rate: {self.learning_rate}")

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
        )

    def _init_history(self):
        """Initialize training history."""
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def _predict(self, data_loader):
        """
        Run inference on a data loader.

        Returns:
            tuple: (labels, predictions, probabilities) as numpy arrays
        """
        self.model.eval()
        all_labels, all_preds, all_proba = [], [], []

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                all_preds.extend(output.argmax(dim=1).cpu().numpy())
                all_proba.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        return np.array(all_labels), np.array(all_preds), np.array(all_proba)

    def _compute_metrics(self, labels, preds, proba):
        """Compute all evaluation metrics."""
        metrics = compute_all_metrics(labels, preds, proba)
        metrics["accuracy"] = accuracy_score(labels, preds)
        metrics["f1_score"] = f1_score(labels, preds, average="weighted", zero_division=0)
        metrics["precision"] = precision_score(labels, preds, average="weighted", zero_division=0)
        metrics["recall"] = recall_score(labels, preds, average="weighted", zero_division=0)
        return metrics

    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_labels.extend(target.cpu().numpy())

        loss = running_loss / len(self.train_loader)
        acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        return loss, acc

    def validate_epoch(self):
        """Run validation and return loss and metrics."""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                loss = self.criterion(self.model(data), target)
                running_loss += loss.item()

        loss = running_loss / len(self.val_loader) if self.val_loader else 0.0
        labels, preds, proba = self._predict(self.val_loader)
        metrics = self._compute_metrics(labels, preds, proba)

        return loss, metrics

    def train(self, epochs=None):
        """
        Train the model.

        Returns:
            dict: Best validation metrics
        """
        epochs = epochs or config.epochs
        print(f"Training {self.model_name} on {self.device}")

        # Minimum threshold: 0.5 = random guessing, require better than random
        min_balanced_acc = 0.55
        best_balanced_acc = 0.0
        best_metrics = {}
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()

            self.scheduler.step(val_loss)
            self._update_history(train_loss, train_acc, val_loss, val_metrics["accuracy"])

            self._print_epoch_summary(epoch, epochs, train_loss, train_acc, val_loss, val_metrics)

            balanced_acc = val_metrics["balanced_accuracy"]

            # Warn if model is not learning (predicting single class)
            if balanced_acc <= 0.5:
                print(f"Warning: Model may be predicting single class (balanced_acc={balanced_acc:.4f})")

            # Model selection: must beat previous best AND minimum threshold
            if balanced_acc > best_balanced_acc and balanced_acc >= min_balanced_acc:
                best_balanced_acc = balanced_acc
                best_metrics = {**val_metrics, "loss": val_loss, "epoch": epoch + 1}
                self.save_model()
                patience_counter = 0
                print(f"New best model saved (balanced_acc: {best_balanced_acc:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= config.patience_limit:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print("-" * 60)

        if best_metrics:
            print_metrics_summary(best_metrics, f"{self.model_name.upper()} Best Validation")
        else:
            print(f"\nWarning: No model saved for {self.model_name} - never exceeded {min_balanced_acc:.0%} balanced accuracy")

        return best_metrics

    def _update_history(self, train_loss, train_acc, val_loss, val_acc):
        """Update training history."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)

    def _print_epoch_summary(self, epoch, epochs, train_loss, train_acc, val_loss, metrics):
        """Print epoch training summary."""
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {metrics['accuracy']:.4f}")
        print(f"  EER: {metrics['eer']:.2f}%, Balanced Acc: {metrics['balanced_accuracy']:.4f}, "
              f"ROC-AUC: {metrics['roc_auc']:.4f}")

    def save_model(self):
        """Save model checkpoint."""
        os.makedirs(config.model_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": {
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses,
                    "train_accuracies": self.train_accuracies,
                    "val_accuracies": self.val_accuracies,
                },
            },
            os.path.join(config.model_dir, f"{self.model_name}_best.pth"),
        )

    def evaluate(self, test_loader, return_predictions=False):
        """
        Evaluate model on a test set.

        Args:
            test_loader: DataLoader for test data
            return_predictions: If True, also return raw predictions for plotting

        Returns:
            dict: Evaluation metrics
            If return_predictions=True, returns (metrics, predictions_dict)
        """
        labels, preds, proba = self._predict(test_loader)
        metrics = self._compute_metrics(labels, preds, proba)

        if return_predictions:
            predictions = {
                "y_true": labels.tolist(),
                "y_pred": preds.tolist(),
                "y_proba": proba.tolist(),
            }
            return metrics, predictions
        return metrics

    def get_training_history(self):
        """
        Get training history for plotting.

        Returns:
            dict: Training history with losses and accuracies per epoch
        """
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }
