"""
Sweep Recovery Script

For each sweep experiment folder:
  1. Reads config.json (model, k_shot, seed)
  2. Loads already-saved seen-class metrics
  3. Re-runs ONLY the zero-day evaluation from the saved checkpoint
  4. Rebuilds sweep_raw_results.csv and sweep_summary.csv

Run AFTER patching ExperimentManager.update_master_table.

Usage:
    python recover_sweep.py
"""

import json
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim

from src.config import config as cfg
from src.data.dataset import CICIoTDataset
from src.data.meta_dataloader import get_zero_day_dataloader
from src.models.meta_factory import get_meta_model
from src.evaluation.meta_evaluator import MetaEvaluator
from src.meta.algorithms.protonet_algorithm import ProtoNetAlgorithm
from src.meta.algorithms.maml_algorithm import MAMLAlgorithm
from src.meta.algorithms.reptile_algorithm import ReptileAlgorithm
from src.utils.seed import set_seed


METRIC_COLUMNS = [
    "accuracy", "precision", "recall", "f1_score",
    "balanced_accuracy", "false_positive_rate",
    "roc_auc", "loss", "inference_time",
]


class RecoveryExperiment:
    """Minimal stand-in for ExperimentManager pointing at an
    EXISTING experiment folder instead of creating a new one."""

    def __init__(self, root):

        self.root = Path(root)

        self.model_name = self.root.name

        self.experiment_name = self.root.name

        self.config_dir = self.root / "config"
        self.metrics_dir = self.root / "metrics"
        self.predictions_dir = self.root / "predictions"
        self.plots_dir = self.root / "plots"
        self.checkpoint_dir = self.root / "checkpoints"

    def update_master_table(self, metrics, model_label=None):

        comparison_file = Path("reports/tables/model_comparison.csv")

        comparison_file.parent.mkdir(parents=True, exist_ok=True)

        row = {"Model": model_label or self.model_name, **metrics}

        new_df = pd.DataFrame([row])

        if comparison_file.exists():

            try:
                old = pd.read_csv(comparison_file)
                df = pd.concat([old, new_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                df = new_df

        else:
            df = new_df

        df.to_csv(comparison_file, index=False)


def build_algorithm(model_name, model):

    # Optimizer needed only to satisfy constructors;
    # zero-day evaluation performs no meta-updates.
    dummy_optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if model_name == "PROTONET":
        return ProtoNetAlgorithm()

    if model_name == "MAML":
        return MAMLAlgorithm(
            meta_optimizer=dummy_optimizer,
            inner_lr=cfg.INNER_LR,
            inner_steps=cfg.INNER_STEPS,
            first_order=True,
            device=cfg.DEVICE,
        )

    if model_name == "REPTILE":
        return ReptileAlgorithm(
            meta_optimizer=dummy_optimizer,
            inner_lr=cfg.REPTILE_INNER_LR,
            inner_steps=cfg.REPTILE_INNER_STEPS,
            device=cfg.DEVICE,
        )

    raise ValueError(f"Unknown model: {model_name}")


def main():

    test_dataset = CICIoTDataset(
        feature_file="X_test.csv",
        label_file="y_test.csv",
    )

    input_dim = test_dataset.X.shape[1]

    results = []

    experiment_dirs = sorted(Path("experiments").iterdir())

    for exp_dir in experiment_dirs:

        config_path = exp_dir / "config" / "config.json"

        if not config_path.exists():
            continue

        with open(config_path) as f:
            run_config = json.load(f)

        # Only recover sweep runs (they logged seed + k_shot)
        if "seed" not in run_config:
            print(f"Skipping (not a sweep run): {exp_dir.name}")
            continue

        model_name = run_config["model"]
        k_shot = run_config["k_shot"]
        seed = run_config["seed"]

        checkpoint = exp_dir / "checkpoints" / f"{model_name.lower()}.pt"

        seen_metrics_path = exp_dir / "metrics" / "test_metrics.csv"

        if not checkpoint.exists() or not seen_metrics_path.exists():
            print(f"Skipping (incomplete): {exp_dir.name}")
            continue

        print("\n" + "=" * 70)
        print(f"Recovering: {exp_dir.name} "
              f"({model_name} | K={k_shot} | seed={seed})")
        print("=" * 70)

        set_seed(seed)

        # ---- seen-class metrics: already on disk ----
        seen_metrics = pd.read_csv(seen_metrics_path).iloc[0].to_dict()

        # ---- zero-day: run now (skip if already present) ----
        zd_metrics_path = exp_dir / "metrics" / "zero_day" / "test_metrics.csv"

        if zd_metrics_path.exists():

            print("Zero-day metrics already exist — reusing.")

            zd_metrics = pd.read_csv(zd_metrics_path).iloc[0].to_dict()

        else:

            zero_day_loader = get_zero_day_dataloader(
                test_dataset=test_dataset,
                n_way=cfg.N_WAY,
                k_shot=k_shot,
                query_size=cfg.QUERY_SIZE,
                episodes=cfg.ZERO_DAY_EPISODES,
                zero_day_classes=cfg.ZERO_DAY_CLASSES,
            )

            model = get_meta_model(model_name, input_dim)

            algorithm = build_algorithm(model_name, model)

            experiment = RecoveryExperiment(exp_dir)

            evaluator = MetaEvaluator(
                model=model,
                algorithm=algorithm,
                episode_loader=zero_day_loader,
                device=cfg.DEVICE,
                checkpoint_path=checkpoint,
                experiment=experiment,
                tag="zero_day",
            )

            zd_metrics = evaluator.evaluate()

        # ---- collect rows ----
        for split, metrics in [("seen", seen_metrics),
                               ("zero_day", zd_metrics)]:

            row = {
                "model": model_name,
                "k_shot": k_shot,
                "seed": seed,
                "split": split,
                "experiment": exp_dir.name,
                "status": "OK",
            }

            for col in METRIC_COLUMNS:
                if col in metrics:
                    row[col] = metrics[col]

            results.append(row)

        # incremental save
        pd.DataFrame(results).to_csv(
            "reports/tables/sweep_raw_results.csv", index=False)

    # ---- aggregate ----
    df = pd.DataFrame(results)

    agg_metrics = [c for c in METRIC_COLUMNS if c in df.columns]

    summary = (
        df.groupby(["model", "k_shot", "split"])[agg_metrics]
        .agg(["mean", "std"])
        .round(4)
    )

    summary.columns = [f"{m}_{s}" for m, s in summary.columns]

    summary = summary.reset_index()

    summary.to_csv("reports/tables/sweep_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("RECOVERY COMPLETE")
    print("=" * 70)
    print(summary[["model", "k_shot", "split",
                   "accuracy_mean", "accuracy_std"]].to_string(index=False))


if __name__ == "__main__":
    main()