"""Evaluation metrics and reporting for spoof detection."""

import json
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    recall_score,
    roc_curve,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from src.config import config


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


def generate_report(val_metrics, test_metrics, total_samples):
    """
    Generate and save a JSON report.

    Args:
        val_metrics: dict mapping model names to validation metrics
        test_metrics: dict mapping model names to test metrics
        total_samples: total number of samples in the dataset
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

    with open(config.report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved: {config.report_file}")
