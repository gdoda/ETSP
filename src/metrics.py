"""
# Recommended Metrics
- EER
- Balanced Accuracy
- ROC AUC
- Per-class Recall
- [Optional]
  - Precision-Recall AUC
  - F1 score for both classes
  - confusion matrix
"""

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    recall_score,
    roc_curve,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def calculate_eer(y_true, y_scores):
    """
    Calculate Equal Error Rate (EER)

    Args:
        y_true: numpy array of ground truth labels (0 or 1)
        y_scores: numpy array of predicted probabilities for positive class

    Returns:
        float: EER as percentage
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100  # Return as percentage


def compute_all_metrics(y_true, y_pred, y_proba):
    """
    Compute all metrics for spoof detection evaluation

    Args:
        y_true: numpy array of ground truth labels (0=bonafide, 1=spoof)
        y_pred: numpy array of predicted labels (0 or 1)
        y_proba: numpy array of predicted probabilities for spoof class (class 1)

    Returns:
        dict: Dictionary containing all computed metrics
    """
    metrics = {}

    # Balanced Accuracy
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    # ROC AUC
    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    # EER (custom implementation)
    metrics["eer"] = calculate_eer(y_true, y_proba)

    # Per-class Recall
    metrics["bonafide_recall"] = recall_score(y_true, y_pred, pos_label=0)
    metrics["spoof_recall"] = recall_score(y_true, y_pred, pos_label=1)

    return metrics


def print_metrics(metrics, prefix=""):
    """
    Pretty print metrics in a readable format

    Args:
        metrics: dict containing metric values
        prefix: string to prepend to output (e.g., "Validation" or "Test")
    """
    print(f"\n{'=' * 50}")
    if prefix:
        print(f"{prefix} Metrics:")
    else:
        print("Metrics:")
    print(f"{'=' * 50}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"  ROC AUC:           {metrics['roc_auc']:.4f}")
    print(f"  EER:               {metrics['eer']:.2f}%")
    print(f"  Bonafide Recall:   {metrics['bonafide_recall']:.4f}")
    print(f"  Spoof Recall:      {metrics['spoof_recall']:.4f}")
    print(f"{'=' * 50}\n")
