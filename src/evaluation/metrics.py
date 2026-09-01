"""
Evaluation Metrics Module

This module provides reusable evaluation functions for all models in the
Meta-Learning IoT Intrusion Detection project.

Supported Metrics
-----------------
- Accuracy
- Precision (Macro)
- Recall (Macro)
- F1 Score (Macro)
- Balanced Accuracy
- Specificity (Macro)
- False Positive Rate (Macro)
- ROC-AUC (Multi-class)
- Confusion Matrix
- Classification Report
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


# ==========================================================
# Compute Main Metrics
# ==========================================================

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Computes all evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    y_prob : ndarray, optional
        Predicted class probabilities.
        Required for ROC-AUC.

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.
    """

    metrics = {}

    metrics["accuracy"] = accuracy_score(
        y_true,
        y_pred,
    )

    metrics["precision"] = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["recall"] = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["f1_score"] = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["balanced_accuracy"] = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    # -----------------------------------------
    # Confusion Matrix
    # -----------------------------------------

    cm = confusion_matrix(y_true, y_pred)

    # -----------------------------------------
    # False Positive Rate
    # -----------------------------------------

    fp_rates = []

    specificities = []

    for i in range(len(cm)):

        tp = cm[i, i]

        fp = cm[:, i].sum() - tp

        fn = cm[i, :].sum() - tp

        tn = cm.sum() - (tp + fp + fn)

        if (fp + tn) == 0:

            fpr = 0

            specificity = 0

        else:

            fpr = fp / (fp + tn)

            specificity = tn / (tn + fp)

        fp_rates.append(fpr)

        specificities.append(specificity)

    metrics["false_positive_rate"] = np.mean(fp_rates)

    metrics["specificity"] = np.mean(specificities)

    # -----------------------------------------
    # ROC AUC
    # -----------------------------------------

    if y_prob is not None:

        try:

            metrics["roc_auc"] = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
            )

        except Exception:

            metrics["roc_auc"] = None

    else:

        metrics["roc_auc"] = None

    return metrics


# ==========================================================
# Confusion Matrix
# ==========================================================

def get_confusion_matrix(
    y_true,
    y_pred,
):

    return confusion_matrix(
        y_true,
        y_pred,
    )


# ==========================================================
# Classification Report
# ==========================================================

def get_classification_report(
    y_true,
    y_pred,
):

    return classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )


# ==========================================================
# Pretty Printing
# ==========================================================

def print_metrics(metrics):
    """
    Nicely prints metrics to the console.
    """

    print("\nEvaluation Metrics")
    print("=" * 50)

    for key, value in metrics.items():

        if value is None:

            print(f"{key:25s}: N/A")

        else:

            print(f"{key:25s}: {value:.4f}")

    print("=" * 50)