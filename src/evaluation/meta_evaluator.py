"""
Meta Evaluator

Evaluates meta-learning models using episodic tasks.

Supports

- ProtoNet
- MAML
- Reptile
"""

import time
import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import (
    compute_metrics,
    get_classification_report,
    get_confusion_matrix,
    print_metrics,
)

from src.visualization.visualizer import Visualizer


class MetaEvaluator:

    def __init__(

        self,

        model,

        algorithm,

        episode_loader,

        device,

        checkpoint_path,

        experiment,

        tag=None,

    ):
        self.model = model

        self.algorithm = algorithm

        self.episode_loader = episode_loader

        self.device = device

        self.checkpoint_path = checkpoint_path

        self.experiment = experiment

        self.tag = tag

        # ---------------------------------------------------
        # Redirect output paths into a subfolder when tagged
        # (e.g. tag="zero_day") so results never overwrite the
        # default seen-class evaluation outputs.
        # ---------------------------------------------------

        if tag:

            self.metrics_dir = self.experiment.metrics_dir / tag

            self.predictions_dir = self.experiment.predictions_dir / tag

            self.plots_dir = self.experiment.plots_dir / tag

        else:

            self.metrics_dir = self.experiment.metrics_dir

            self.predictions_dir = self.experiment.predictions_dir

            self.plots_dir = self.experiment.plots_dir

        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # Load Model
    # =====================================================

    def load_model(self):

        self.model.load_state_dict(

            torch.load(

                self.checkpoint_path,

                map_location=self.device,

            )

        )

        self.model.to(self.device)

        self.model.eval()

        print("Best model loaded.")

    # =====================================================
    # Predict
    # =====================================================

    def predict(self):

        y_true = []

        y_pred = []

        probabilities = []

        adaptation_times = []          # <- with the other accumulators

        total_loss = 0

        start = time.time()

        for episode in self.episode_loader:

            (
                support_x,
                support_y,
                query_x,
                query_y,
            ) = episode

            # ---------------------------------------------
            # Remove Episode Dimension
            # ---------------------------------------------

            support_x = support_x.squeeze(0).to(self.device)

            support_y = support_y.squeeze(0).to(self.device)

            query_x = query_x.squeeze(0).to(self.device)

            query_y = query_y.squeeze(0).to(self.device)

            # ---------------------------------------------
            # Meta-Learning Validation
            # ---------------------------------------------

            result = self.algorithm.validation_step(

                self.model,

                support_x,

                support_y,

                query_x,

                query_y,

            )
            adaptation_times.append(result.get("adaptation_time", 0.0))

            total_loss += result["loss"].item()

            logits = result["logits"]

            preds = result["predictions"]

            probs = torch.softmax(

                logits,

                dim=1,

            )

            y_true.extend(

                query_y.detach().cpu().numpy()


            )

            y_pred.extend(

                preds.detach().cpu().numpy()

            )

            probabilities.extend(

                probs.detach().cpu().numpy()

            )

        inference_time = time.time() - start

        return (
            np.array(y_true),
            np.array(y_pred),
            np.array(probabilities),
            total_loss / len(self.episode_loader),
            inference_time,
            float(np.mean(adaptation_times)),
        )
        # =====================================================
    # Evaluate
    # =====================================================

    def evaluate(self):

        # ---------------------------------------------
        # Load Best Model
        # ---------------------------------------------

        self.load_model()

        # ---------------------------------------------
        # Predict
        # ---------------------------------------------

        (
            y_true,
            y_pred,
            y_prob,
            loss,
            inference_time,
            adaptation_time,
        ) = self.predict()

        # ---------------------------------------------
        # Compute Metrics
        # ---------------------------------------------

        metrics = compute_metrics(

            y_true,

            y_pred,

            y_prob,

        )

        metrics["loss"] = loss

        metrics["inference_time"] = inference_time

        metrics["adaptation_time"] = adaptation_time

        print_metrics(metrics)

        # ---------------------------------------------
        # Save Metrics
        # ---------------------------------------------

        metrics_df = pd.DataFrame([metrics])

        metrics_df.to_csv(

            self.metrics_dir /
            "test_metrics.csv",

            index=False,

        )

        # ---------------------------------------------
        # Classification Report
        # ---------------------------------------------

        report = get_classification_report(

            y_true,

            y_pred,

        )

        report_df = pd.DataFrame(report).transpose()

        report_df.to_csv(

            self.metrics_dir /
            "classification_report.csv",

            index=True,

        )

        # ---------------------------------------------
        # Save Predictions
        # ---------------------------------------------

        prediction_df = pd.DataFrame({

            "Actual": y_true,

            "Predicted": y_pred,

        })

        prediction_df.to_csv(

            self.predictions_dir /
            "test_predictions.csv",

            index=False,

        )

        # ---------------------------------------------
        # Confusion Matrix
        # ---------------------------------------------

        cm = get_confusion_matrix(

            y_true,

            y_pred,

        )

        # ---------------------------------------------
        # Visualisations
        # ---------------------------------------------

        visualizer = Visualizer(

            self.plots_dir

        )

        visualizer.plot_confusion_matrix(

            cm,

            np.unique(y_true),

        )

        visualizer.plot_metric_bar(

            metrics,

        )

        visualizer.plot_roc_curve(

            y_true,

            y_prob,

        )

        visualizer.plot_precision_recall_curve(

            y_true,

            y_prob,

        )

        visualizer.plot_prediction_distribution(

            y_pred,

        )

        visualizer.plot_class_accuracy(

            report,

        )

        # ---------------------------------------------
        # Update Experiment
        # ---------------------------------------------

        model_label = (

            f"{self.experiment.model_name}_{self.tag}"
            if self.tag else None

        )

        self.experiment.update_master_table(

            metrics,

            model_label=model_label,

        )

        print("\n" + "=" * 70)

        if self.tag:

            print(f"META EVALUATION COMPLETED [{self.tag.upper()}]")

        else:

            print("META EVALUATION COMPLETED")

        print("=" * 70)

        return metrics