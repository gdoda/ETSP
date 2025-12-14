import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

from src.config import config


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, model_name="cnn", use_class_weights=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name

        self.device = config.device
        self.model.to(self.device)

        # Use class weights for imbalanced dataset
        if use_class_weights:
            class_weights = config.class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using class weights: {config.class_weights.tolist()}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=config.scheduler_patience,
            factor=config.scheduler_factor
        )

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = (
            accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        )
        return epoch_loss, epoch_acc

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        epoch_loss = (
            running_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        )
        epoch_acc = (
            accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        )

        # Calculate additional metrics
        if len(all_labels) > 0:
            precision = precision_score(all_labels, all_preds, average="weighted")
            recall = recall_score(all_labels, all_preds, average="weighted")
            f1 = f1_score(all_labels, all_preds, average="weighted")
        else:
            precision = recall = f1 = 0.0

        return epoch_loss, epoch_acc, precision, recall, f1

    def train(self, epochs=None):
        epochs = epochs or config.epochs
        print(f"Training {self.model_name} on {self.device}")

        best_val_acc = 0.0
        best_metrics = {}
        patience_counter = 0
        patience_limit = config.patience_limit

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, precision, recall, f1 = self.validate_epoch()

            self.scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                # Store best metrics
                best_metrics = {
                    "accuracy": val_acc,
                    "loss": val_loss,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "epoch": epoch + 1
                }
                
                self.save_model()
                patience_counter = 0
                print(f"New best model saved with val_acc: {val_acc:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print("-" * 60)
        
        return best_metrics

    def save_model(self):
        os.makedirs(config.model_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_accuracies": self.train_accuracies,
                "val_accuracies": self.val_accuracies,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            os.path.join(config.model_dir, f"{self.model_name}_best.pth"),
        )
