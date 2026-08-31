"""
Visualization Module

Generates publication-quality figures for the dissertation.

Supported Plots
---------------
1. Training Loss Curve
2. Validation Accuracy Curve
3. Precision / Recall / F1 Curve
4. Confusion Matrix
5. Metrics Bar Chart
6. ROC Curve
7. Precision-Recall Curve
8. Prediction Distribution
9. Per-Class Accuracy
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
)

from sklearn.preprocessing import label_binarize


sns.set_theme(style="whitegrid")




class Visualizer:

    def __init__(self, plot_dir, dpi=300):

        self.plot_dir = Path(plot_dir)

        self.plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.dpi = dpi

    # ==========================================================
    # Training Loss
    # ==========================================================

    def plot_loss(self, history):

        plt.figure(figsize=(8,5))

        plt.plot(
            history["epoch"],
            history["train_loss"],
            label="Training Loss",
            linewidth=2,
        )

        plt.plot(
            history["epoch"],
            history["val_loss"],
            label="Validation Loss",
            linewidth=2,
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            self.plot_dir / "loss_curve.png",
            dpi=self.dpi,
        )

        plt.close()

    # ==========================================================
    # Accuracy
    # ==========================================================

    def plot_accuracy(self, history):

        plt.figure(figsize=(8,5))

        plt.plot(
            history["epoch"],
            history["accuracy"],
            linewidth=2,
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")

        plt.tight_layout()

        plt.savefig(
            self.plot_dir / "accuracy_curve.png",
            dpi=self.dpi,
        )

        plt.close()

    # ==========================================================
    # Precision Recall F1
    # ==========================================================

    def plot_metrics(self, history):

        plt.figure(figsize=(8,5))

        plt.plot(
            history["epoch"],
            history["precision"],
            label="Precision",
            linewidth=2,
        )

        plt.plot(
            history["epoch"],
            history["recall"],
            label="Recall",
            linewidth=2,
        )

        plt.plot(
            history["epoch"],
            history["f1_score"],
            label="F1 Score",
            linewidth=2,
        )

        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            self.plot_dir / "metrics_curve.png",
            dpi=self.dpi,
        )

        plt.close()

    # ==========================================================
    # Confusion Matrix
    # ==========================================================

    def plot_confusion_matrix(
        self,
        confusion_matrix,
        class_names,
    ):

        plt.figure(figsize=(14,12))

        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.savefig(
            self.plot_dir / "confusion_matrix.png",
            dpi=self.dpi,
        )

        plt.close()

    # ==========================================================
    # Metrics Bar Chart
    # ==========================================================

    def plot_metric_bar(self, metrics):

        ignore = ["roc_auc"]

        metric_names = [
            k for k in metrics.keys()
            if k not in ignore and metrics[k] is not None
        ]

        values = [
            metrics[k]
            for k in metric_names
        ]

        plt.figure(figsize=(9,5))

        plt.bar(metric_names, values)

        plt.xticks(rotation=30)

        plt.ylabel("Score")

        plt.title("Evaluation Metrics")

        plt.tight_layout()

        plt.savefig(
            self.plot_dir / "metrics_bar.png",
            dpi=self.dpi,
        )

        plt.close()

    # ==========================================================
    # ROC Curve
    # ==========================================================

    def plot_roc_curve(
        self,
        y_true,
        y_prob,
    ):

        classes = np.unique(y_true)

        y_true_bin = label_binarize(
            y_true,
            classes=classes,
        )

        plt.figure(figsize=(7,6))

        for i in range(len(classes)):

            fpr, tpr, _ = roc_curve(
                y_true_bin[:, i],
                y_prob[:, i],
            )

            roc_auc = auc(
                fpr,
                tpr,
            )

            plt.plot(
                fpr,
                tpr,
                label=f"Class {classes[i]} (AUC={roc_auc:.2f})"
            )

        plt.plot(
            [0,1],
            [0,1],
            linestyle="--",
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.legend(fontsize=8)

        plt.tight_layout()

        plt.savefig(
            self.plot_dir / "roc_curve.png",
            dpi=self.dpi,
        )

        plt.close()

    # ==========================================================
    # Precision Recall Curve
    # ==========================================================

    def plot_precision_recall_curve(
        self,
        y_true,
        y_prob,
    ):

        classes = np.unique(y_true)

        y_true_bin = label_binarize(
            y_true,
            classes=classes,
        )

        plt.figure(figsize=(7,6))

        for i in range(len(classes)):

            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i],
                y_prob[:, i],
            )

            plt.plot(
                recall,
                precision,
                label=f"Class {classes[i]}"
            )

        plt.xlabel("Recall")

        plt.ylabel("Precision")

        plt.title("Precision-Recall Curve")

        plt.legend(fontsize=8)

        plt.tight_layout()

        plt.savefig(
            self.plot_dir / "precision_recall_curve.png",
            dpi=self.dpi,
        )

        plt.close()

    # ==========================================================
    # Prediction Distribution
    # ==========================================================

    def plot_prediction_distribution(
        self,
        predictions,
    ):

        plt.figure(figsize=(10,5))

        sns.countplot(
            x=predictions,
        )

        plt.xlabel("Predicted Class")

        plt.ylabel("Count")

        plt.title("Prediction Distribution")

        plt.tight_layout()

        plt.savefig(
            self.plot_dir / "prediction_distribution.png",
            dpi=self.dpi,
        )

        plt.close()

    # ==========================================================
    # Per-Class Accuracy
    # ==========================================================

    def plot_class_accuracy(
        self,
        report,
    ):

        report_df = pd.DataFrame(report).transpose()

        report_df = report_df.iloc[:-3]

        plt.figure(figsize=(10,6))

        sns.barplot(
            x=report_df.index,
            y=report_df["recall"],
        )

        plt.xticks(rotation=90)

        plt.ylabel("Recall")

        plt.xlabel("Class")

        plt.title("Per-Class Detection Accuracy")

        plt.tight_layout()

        plt.savefig(
            self.plot_dir / "class_accuracy.png",
            dpi=self.dpi,
        )

        plt.close()

    # ==========================================================
    # Load History
    # ==========================================================

    @staticmethod
    def load_history(path):

        return pd.read_csv(path)