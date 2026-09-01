"""
Evaluation Pipeline

This module evaluates a trained model on the test dataset.

Responsibilities
----------------
1. Load trained model
2. Run inference
3. Compute metrics
4. Save predictions
5. Save metrics
6. Generate visualisations
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from src.experiments.experiment_manager import ExperimentManager

import torch

from src.evaluation.metrics import (
    compute_metrics,
    get_confusion_matrix,
    get_classification_report,
    print_metrics,
)

from src.visualization.visualizer import Visualizer


class Evaluator:

    def __init__(
        self,
        model,
        test_loader,
        device,
        checkpoint_path,
        experiment,
        model_name="MLP",
    ):

        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name
        self.experiment = experiment
        self.checkpoint_path = checkpoint_path

        Path("results").mkdir(exist_ok=True)
        Path("results/metrics").mkdir(parents=True, exist_ok=True)
        Path("results/predictions").mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # Load Trained Model
    # ==========================================================

    def load_model(self):

        self.model.load_state_dict(

            torch.load(
                self.checkpoint_path,
                map_location=self.device
            )

        )

        self.model.to(self.device)

        self.model.eval()

        print("Model loaded successfully.")

    # ==========================================================
    # Predict Test Set
    # ==========================================================

    def predict(self):

        predictions = []

        probabilities = []

        labels = []

        start = time.time()

        with torch.no_grad():

            for X, y in self.test_loader:

                X = X.to(self.device)

                y = y.to(self.device)

                outputs = self.model(X)

                probs = torch.softmax(outputs, dim=1)

                preds = torch.argmax(outputs, dim=1)

                predictions.extend(
                    preds.cpu().numpy()
                )

                probabilities.extend(
                    probs.cpu().numpy()
                )

                labels.extend(
                    y.cpu().numpy()
                )

        end = time.time()

        inference_time = end - start

        print(f"Inference Time : {inference_time:.2f} seconds")

        return (

            np.array(labels),

            np.array(predictions),

            np.array(probabilities),

            inference_time,

        )

    # ==========================================================
    # Complete Evaluation
    # ==========================================================
    # ==========================================================
    # Complete Evaluation
    # ==========================================================

    def evaluate(self):

        # ------------------------------------------------------
        # Load Best Model
        # ------------------------------------------------------

        self.load_model()

        # ------------------------------------------------------
        # Run Inference
        # ------------------------------------------------------

        y_true, y_pred, y_prob, inference_time = self.predict()

        # ------------------------------------------------------
        # Compute Metrics
        # ------------------------------------------------------

        metrics = compute_metrics(
            y_true,
            y_pred,
            y_prob,
        )

        metrics["inference_time"] = inference_time

        print_metrics(metrics)

        # ------------------------------------------------------
        # Save Metrics
        # ------------------------------------------------------

        metrics_df = pd.DataFrame([metrics])

        metrics_df.to_csv(
            self.experiment.metrics_dir / "test_metrics.csv",
            index=False,
        )
        
        # ------------------------------------------------------
        # Classification Report
        # ------------------------------------------------------

        report = get_classification_report(
            y_true,
            y_pred,
        )

        report_df = pd.DataFrame(report).transpose()

        report_df.to_csv(
            self.experiment.metrics_dir / "classification_report.csv",
            index=True,
        )

        # ------------------------------------------------------
        # Save Predictions
        # ------------------------------------------------------

        prediction_df = pd.DataFrame({
            "Actual": y_true,
            "Predicted": y_pred,
        })

        prediction_df.to_csv(
            self.experiment.predictions_dir / "test_predictions.csv",
            index=False,
        )
        # ------------------------------------------------------
        # Confusion Matrix
        # ------------------------------------------------------

        cm = get_confusion_matrix(
            y_true,
            y_pred,
        )

        # ------------------------------------------------------
        # Generate Visualisations
        # ------------------------------------------------------

        print("\nGenerating evaluation visualisations...")

        visualizer = Visualizer(
            self.experiment.plots_dir
        )

        # Confusion Matrix
        visualizer.plot_confusion_matrix(
            cm,
            np.unique(y_true),
        )

        # Overall Metrics
        visualizer.plot_metric_bar(
            metrics,
        )

        # ROC Curve
        visualizer.plot_roc_curve(
            y_true,
            y_prob,
        )

        # Precision-Recall Curve
        visualizer.plot_precision_recall_curve(
            y_true,
            y_prob,
        )

        # Prediction Distribution
        visualizer.plot_prediction_distribution(
            y_pred,
        )

        # Per-Class Accuracy
        visualizer.plot_class_accuracy(
            report,
        )

        print("✓ All evaluation plots generated successfully.")

        # ------------------------------------------------------
        # Finish
        # ------------------------------------------------------

        print("\n" + "=" * 70)
        print("Evaluation Completed Successfully")
        print("=" * 70)

        print(f"Accuracy      : {metrics['accuracy']:.4f}")
        print(f"F1 Score      : {metrics['f1_score']:.4f}")
        print(f"ROC-AUC       : {metrics['roc_auc']}")
        print(f"Inference Time: {metrics['inference_time']:.4f} seconds")

        print("\nResults saved to:")
        print("   results/metrics/")
        print("   results/predictions/")
        print("   results/plots/")

        # -----------------------------------------
        # Update Experiment
        # -----------------------------------------

        self.experiment.save_metrics(
            metrics
        )

        self.experiment.save_predictions(
            prediction_df
        )

        self.experiment.update_master_table(
            metrics
        )

        # self.experiment.copy_plots()

        self.experiment.summary()

        return metrics
    
        
    
