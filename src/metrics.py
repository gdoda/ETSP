"""Evaluation metrics and reporting for spoof detection."""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    recall_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from src.config import config

# Plot styling
COLORS = {"CNN": "#2ecc71", "ViT": "#3498db", "RNN": "#e74c3c"}
plt.style.use("seaborn-v0_8-whitegrid")


def calculate_eer(y_true, y_scores):
    """
    Calculate Equal Error Rate (EER).

    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Predicted probabilities for positive class

    Returns:
        EER as percentage
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100


def compute_all_metrics(y_true, y_pred, y_proba):
    """
    Compute all spoof detection metrics.

    Args:
        y_true: Ground truth labels (0=bonafide, 1=spoof)
        y_pred: Predicted labels
        y_proba: Predicted probabilities for spoof class

    Returns:
        Dictionary with all metrics
    """
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "eer": calculate_eer(y_true, y_proba),
        "bonafide_recall": recall_score(y_true, y_pred, pos_label=0),
        "spoof_recall": recall_score(y_true, y_pred, pos_label=1),
    }


def print_metrics_summary(metrics, title=""):
    """Print a concise metrics summary."""
    print(f"\n{'=' * 50}")
    print(f"  {title}" if title else "  Metrics Summary")
    print(f"{'=' * 50}")
    print(f"  Accuracy:          {metrics.get('accuracy', 0):.4f}")
    print(f"  F1-Score:          {metrics.get('f1_score', 0):.4f}")
    print(f"  EER:               {metrics.get('eer', 0):.2f}%")
    print(f"  Balanced Acc:      {metrics.get('balanced_accuracy', 0):.4f}")
    print(f"  ROC AUC:           {metrics.get('roc_auc', 0):.4f}")
    print(f"  Bonafide Recall:   {metrics.get('bonafide_recall', 0):.4f}")
    print(f"  Spoof Recall:      {metrics.get('spoof_recall', 0):.4f}")
    if "epoch" in metrics:
        print(f"  Best Epoch:        {metrics['epoch']}")
    print(f"{'=' * 50}\n")


def print_model_comparison(model_metrics):
    """
    Print a comparison table of metrics for multiple models.

    Args:
        model_metrics: dict mapping model names to their metrics dicts
    """
    print("\n" + "=" * 70)
    print("                    MODEL COMPARISON (Test Set)")
    print("=" * 70)

    model_names = list(model_metrics.keys())
    header = f"{'Metric':<20}" + "".join(f"{name:>14}" for name in model_names)
    print(header)
    print("-" * 70)

    metrics_config = [
        ("EER (%)", "eer", "{:.2f}"),
        ("Balanced Acc", "balanced_accuracy", "{:.4f}"),
        ("ROC AUC", "roc_auc", "{:.4f}"),
        ("Accuracy", "accuracy", "{:.4f}"),
        ("F1-Score", "f1_score", "{:.4f}"),
        ("Bonafide Recall", "bonafide_recall", "{:.4f}"),
        ("Spoof Recall", "spoof_recall", "{:.4f}"),
    ]

    for label, key, fmt in metrics_config:
        row = f"{label:<20}"
        for name in model_names:
            value = model_metrics[name].get(key, 0)
            row += f"{fmt.format(value):>14}"
        print(row)

    print("=" * 70 + "\n")


def generate_report(val_metrics, test_metrics, total_samples,
                    training_history=None, predictions=None):
    """
    Generate and save a JSON report.

    Args:
        val_metrics: dict mapping model names to validation metrics
        test_metrics: dict mapping model names to test metrics
        total_samples: total number of samples in the dataset
        training_history: dict mapping model names to training history (optional)
        predictions: dict mapping model names to test predictions (optional)
    """

    def format_model_metrics(val_m, test_m):
        """Format metrics for a single model."""
        return {
            "validation": {
                "best_epoch": val_m.get("epoch", 0),
                "accuracy": f"{val_m.get('accuracy', 0) * 100:.2f}%",
                "loss": f"{val_m.get('loss', 0):.4f}",
                "f1_score": f"{val_m.get('f1_score', 0):.4f}",
                "eer": f"{val_m.get('eer', 0):.2f}%",
                "balanced_accuracy": f"{val_m.get('balanced_accuracy', 0):.4f}",
                "roc_auc": f"{val_m.get('roc_auc', 0):.4f}",
                "bonafide_recall": f"{val_m.get('bonafide_recall', 0):.4f}",
                "spoof_recall": f"{val_m.get('spoof_recall', 0):.4f}",
            },
            "test": {
                "accuracy": f"{test_m.get('accuracy', 0) * 100:.2f}%",
                "f1_score": f"{test_m.get('f1_score', 0):.4f}",
                "eer": f"{test_m.get('eer', 0):.2f}%",
                "balanced_accuracy": f"{test_m.get('balanced_accuracy', 0):.4f}",
                "roc_auc": f"{test_m.get('roc_auc', 0):.4f}",
                "bonafide_recall": f"{test_m.get('bonafide_recall', 0):.4f}",
                "spoof_recall": f"{test_m.get('spoof_recall', 0):.4f}",
            },
        }

    report = {
        "project_title": "Synthetic Voice Detection System",
        "dataset": {
            "name": "ASVspoof2021 LA eval",
            "total_samples": total_samples,
            "split_ratios": {
                "train": config.train_ratio,
                "val": config.val_ratio,
                "test": config.test_ratio,
            },
        },
        "training_config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "early_stopping_patience": config.patience_limit,
            "class_weights": config.class_weights.tolist(),
            "model_selection_metric": "balanced_accuracy",
        },
        "model_performance": {
            name: format_model_metrics(val_metrics[name], test_metrics[name])
            for name in val_metrics.keys()
        },
        "metrics_description": {
            "eer": "Equal Error Rate - lower is better",
            "balanced_accuracy": "Average of per-class recalls",
            "roc_auc": "Area under ROC curve - higher is better",
            "bonafide_recall": "True positive rate for real speech",
            "spoof_recall": "True positive rate for fake speech",
        },
    }

    # Add training history for plotting learning curves
    if training_history:
        report["training_history"] = training_history

    # Add predictions for ROC curves and confusion matrices
    if predictions:
        report["predictions"] = predictions

    with open(config.report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved: {config.report_file}")


def plot_training_curves(training_history, save_path=None):
    """
    Plot training and validation curves for all models.

    Args:
        training_history: dict mapping model names to their training history
                         Each history contains: train_losses, val_losses,
                         train_accuracies, val_accuracies
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, len(training_history), figsize=(5 * len(training_history), 8))

    if len(training_history) == 1:
        axes = axes.reshape(-1, 1)

    for idx, (name, history) in enumerate(training_history.items()):
        epochs = range(1, len(history["train_losses"]) + 1)
        color = COLORS.get(name, "#333333")

        # Loss plot
        ax_loss = axes[0, idx]
        ax_loss.plot(epochs, history["train_losses"], "-", color=color, label="Train", linewidth=2)
        ax_loss.plot(epochs, history["val_losses"], "--", color=color, label="Val", linewidth=2)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"{name} - Loss")
        ax_loss.legend()
        ax_loss.set_xticks(epochs)

        # Accuracy plot
        ax_acc = axes[1, idx]
        ax_acc.plot(epochs, history["train_accuracies"], "-", color=color, label="Train", linewidth=2)
        ax_acc.plot(epochs, history["val_accuracies"], "--", color=color, label="Val", linewidth=2)
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title(f"{name} - Accuracy")
        ax_acc.legend()
        ax_acc.set_ylim(0, 1)
        ax_acc.set_xticks(epochs)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved: {save_path}")

    return fig


def plot_roc_curves(predictions, save_path=None):
    """
    Plot ROC curves for all models.

    Args:
        predictions: dict mapping model names to their predictions
                    Each contains: y_true, y_proba
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, preds in predictions.items():
        y_true = np.array(preds["y_true"])
        y_proba = np.array(preds["y_proba"])

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        eer = calculate_eer(y_true, y_proba)

        color = COLORS.get(name, "#333333")
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={auc:.3f}, EER={eer:.1f}%)")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Model Comparison")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curves saved: {save_path}")

    return fig


def plot_confusion_matrices(predictions, save_path=None):
    """
    Plot confusion matrices for all models.

    Args:
        predictions: dict mapping model names to their predictions
                    Each contains: y_true, y_pred
        save_path: Optional path to save the figure
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    class_names = ["Bonafide", "Spoof"]

    for idx, (name, preds) in enumerate(predictions.items()):
        y_true = np.array(preds["y_true"])
        y_pred = np.array(preds["y_pred"])

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
        axes[idx].set_title(f"{name}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrices saved: {save_path}")

    return fig


def plot_metric_comparison(test_metrics, save_path=None):
    """
    Plot bar chart comparison of key metrics across models.

    Args:
        test_metrics: dict mapping model names to their test metrics
        save_path: Optional path to save the figure
    """
    metrics_to_plot = [
        ("balanced_accuracy", "Balanced Accuracy", True),
        ("roc_auc", "ROC AUC", True),
        ("eer", "EER (%)", False),  # Lower is better
        ("bonafide_recall", "Bonafide Recall", True),
        ("spoof_recall", "Spoof Recall", True),
    ]

    model_names = list(test_metrics.keys())
    n_metrics = len(metrics_to_plot)
    x = np.arange(len(model_names))
    width = 0.7

    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 4))

    for idx, (metric_key, metric_label, higher_better) in enumerate(metrics_to_plot):
        values = [test_metrics[name].get(metric_key, 0) for name in model_names]
        colors = [COLORS.get(name, "#333333") for name in model_names]

        bars = axes[idx].bar(x, values, width, color=colors)
        axes[idx].set_ylabel(metric_label)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(model_names)
        axes[idx].set_title(metric_label)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].annotate(
                f"{val:.2f}" if metric_key != "eer" else f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center", va="bottom", fontsize=9
            )

        # Set appropriate y limits
        if metric_key == "eer":
            axes[idx].set_ylim(0, max(values) * 1.2 if max(values) > 0 else 50)
        else:
            axes[idx].set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Metric comparison saved: {save_path}")

    return fig


def generate_all_plots(report_path=None, output_dir=None):
    """
    Generate all plots from saved report JSON.

    Args:
        report_path: Path to project_report.json (defaults to config.report_file)
        output_dir: Directory to save plots (defaults to config.model_dir)

    Returns:
        dict of figure objects
    """
    report_path = report_path or config.report_file
    output_dir = output_dir or config.model_dir

    with open(report_path, "r") as f:
        report = json.load(f)

    figures = {}

    # Training curves
    if "training_history" in report:
        figures["training_curves"] = plot_training_curves(
            report["training_history"],
            save_path=f"{output_dir}/training_curves.png"
        )

    # ROC curves and confusion matrices
    if "predictions" in report:
        figures["roc_curves"] = plot_roc_curves(
            report["predictions"],
            save_path=f"{output_dir}/roc_curves.png"
        )
        figures["confusion_matrices"] = plot_confusion_matrices(
            report["predictions"],
            save_path=f"{output_dir}/confusion_matrices.png"
        )

    # Metric comparison (reconstruct from model_performance)
    test_metrics = {}
    for name, perf in report["model_performance"].items():
        test_metrics[name] = {
            "balanced_accuracy": float(perf["test"]["balanced_accuracy"]),
            "roc_auc": float(perf["test"]["roc_auc"]),
            "eer": float(perf["test"]["eer"].replace("%", "")),
            "bonafide_recall": float(perf["test"]["bonafide_recall"]),
            "spoof_recall": float(perf["test"]["spoof_recall"]),
        }

    figures["metric_comparison"] = plot_metric_comparison(
        test_metrics,
        save_path=f"{output_dir}/metric_comparison.png"
    )

    print(f"\nAll plots saved to: {output_dir}/")
    return figures
