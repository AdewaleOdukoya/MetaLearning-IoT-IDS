"""
Transfer-Learning Baseline — Episodic Evaluation

Evaluates the pretrained fine-tune baseline over the same
k_shot x seed grid, on the same seen-class and zero-day episode
distributions, as the meta-learning sweep. APPENDS results to
sweep_raw_results.csv (after a timestamped backup) and regenerates
sweep_summary.csv.

Prerequisite: python -m src.training.pretrain_finetune

Usage:
    python evaluate_finetune.py
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import config as cfg
from src.data.dataset import CICIoTDataset
from src.data.episodes import EpisodeDataset
from src.data.meta_dataloader import get_zero_day_dataloader
from src.models.mlp import MLP
from src.evaluation.meta_evaluator import MetaEvaluator
from src.meta.algorithms.finetune_algorithm import FinetuneBaselineAlgorithm
from src.experiments.experiment_manager import ExperimentManager
from src.utils.seed import set_seed


K_SHOTS = [1, 5, 10]

SEEDS = [42, 43, 44, 45, 46]

METRIC_COLUMNS = [
    "accuracy", "precision", "recall", "f1_score",
    "balanced_accuracy", "false_positive_rate",
    "roc_auc", "loss", "inference_time", "adaptation_time",
]

RAW_PATH = Path("reports/tables/sweep_raw_results.csv")


def backup(path):

    if path.exists():

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        dst = path.with_name(f"{path.stem}_backup_{stamp}{path.suffix}")

        shutil.copy2(path, dst)

        print(f"Backed up {path.name} -> {dst.name}")


def main():

    meta_path = Path("results/pretrained/finetune_meta.json")

    checkpoint = Path("results/pretrained/finetune_mlp.pt")

    if not meta_path.exists() or not checkpoint.exists():

        raise FileNotFoundError(
            "Pretrained baseline not found. Run "
            "`python -m src.training.pretrain_finetune` first.")

    with open(meta_path) as f:

        info = json.load(f)

    meta_train_classes = np.array(info["meta_train_classes"])

    val_dataset = CICIoTDataset(
        feature_file="X_validation.csv", label_file="y_validation.csv")

    test_dataset = CICIoTDataset(
        feature_file="X_test.csv", label_file="y_test.csv")

    rows = []

    for k_shot in K_SHOTS:

        for seed in SEEDS:

            print("\n" + "#" * 70)
            print(f"# FINETUNE baseline | {k_shot}-shot | seed {seed}")
            print("#" * 70)

            set_seed(seed)

            # Seen-class episodes (same distribution the meta
            # models were validated on)
            seen_loader = DataLoader(
                EpisodeDataset(
                    dataset=val_dataset,
                    n_way=cfg.N_WAY,
                    k_shot=k_shot,
                    query_size=cfg.QUERY_SIZE,
                    episodes=max(cfg.EPISODES // 5, 100),
                    allowed_classes=meta_train_classes,
                ),
                batch_size=1, shuffle=False,
            )

            zero_day_loader = get_zero_day_dataloader(
                test_dataset=test_dataset,
                n_way=cfg.N_WAY,
                k_shot=k_shot,
                query_size=cfg.QUERY_SIZE,
                episodes=cfg.ZERO_DAY_EPISODES,
                zero_day_classes=cfg.ZERO_DAY_CLASSES,
            )

            model = MLP(
                input_dim=info["input_dim"],
                num_classes=info["num_classes"],
            )

            algorithm = FinetuneBaselineAlgorithm(
                finetune_lr=cfg.INNER_LR,
                finetune_steps=20,
                device=cfg.DEVICE,
            )

            experiment = ExperimentManager(
                f"FINETUNE_K{k_shot}_S{seed}")

            experiment.create_readme()

            experiment.save_config({
                "model": "FINETUNE",
                "k_shot": k_shot,
                "seed": seed,
                "n_way": cfg.N_WAY,
                "query_size": cfg.QUERY_SIZE,
                "finetune_lr": cfg.INNER_LR,
                "finetune_steps": 20,
                "pretrained_checkpoint": str(checkpoint),
                "zero_day_classes": list(cfg.ZERO_DAY_CLASSES),
            })

            for split, loader, tag in [
                ("seen", seen_loader, None),
                ("zero_day", zero_day_loader, "zero_day"),
            ]:

                evaluator = MetaEvaluator(
                    model=model,
                    algorithm=algorithm,
                    episode_loader=loader,
                    device=cfg.DEVICE,
                    checkpoint_path=checkpoint,
                    experiment=experiment,
                    tag=tag,
                )

                metrics = evaluator.evaluate()

                row = {
                    "model": "FINETUNE",
                    "k_shot": k_shot,
                    "seed": seed,
                    "split": split,
                    "experiment": experiment.experiment_name,
                    "status": "OK",
                }

                for col in METRIC_COLUMNS:

                    if col in metrics:

                        row[col] = metrics[col]

                rows.append(row)

    # ---------- append (never overwrite) ----------

    backup(RAW_PATH)

    new_df = pd.DataFrame(rows)

    if RAW_PATH.exists():

        old = pd.read_csv(RAW_PATH)

        # drop any previous FINETUNE rows so re-runs don't duplicate
        old = old[old["model"] != "FINETUNE"]

        combined = pd.concat([old, new_df], ignore_index=True)

    else:

        combined = new_df

    combined.to_csv(RAW_PATH, index=False)

    # ---------- regenerate summary ----------

    ok = combined[combined["status"] == "OK"]

    agg = [c for c in METRIC_COLUMNS if c in ok.columns]

    summary = (
        ok.groupby(["model", "k_shot", "split"])[agg]
        .agg(["mean", "std"]).round(4)
    )

    summary.columns = [f"{m}_{s}" for m, s in summary.columns]

    summary.reset_index().to_csv(
        "reports/tables/sweep_summary.csv", index=False)

    print("\nFINETUNE baseline results appended. Summary regenerated.")


if __name__ == "__main__":

    main()