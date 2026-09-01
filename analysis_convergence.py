"""
Convergence Comparison Figure (Objective 4 evidence)

Plots validation accuracy (and loss) vs. epoch for ProtoNet, MAML,
and Reptile on the SAME axes, using one representative run per
model (same k_shot and seed, for a fair comparison) — this shows
convergence SPEED differences directly, which the summary tables
alone don't capture (they only show the final/best accuracy).

Reads training_history.csv from each model's experiment folder
under experiments/, matched by a filename pattern.

Usage:
    python analysis_convergence.py
    python analysis_convergence.py --dataset CICIOT --k-shot 5 --seed 42
    python analysis_convergence.py --dataset TONIOT --k-shot 5 --seed 42 --prefix toniot_
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

FIG_DIR = Path("reports/figures")

FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

MODELS = ["PROTONET", "MAML", "REPTILE"]

MODEL_LABELS = {
    "PROTONET": "Prototypical Networks",
    "MAML": "MAML (First-Order)",
    "REPTILE": "Reptile",
}

PALETTE = {
    "PROTONET": "#1f77b4",
    "MAML": "#d62728",
    "REPTILE": "#2ca02c",
}


def find_history(dataset, model, k_shot, seed):
    """
    Locates training_history.csv for a specific (dataset, model,
    k_shot, seed) combination by matching the experiment folder
    naming pattern used by run_experiments.py / train_meta.py.
    """

    if dataset == "TONIOT":
        pattern = f"TONIOT_{model}_K{k_shot}_S{seed}_*"
    else:
        pattern = f"{model}_K{k_shot}_S{seed}_*"

    matches = sorted(Path("experiments").glob(pattern))

    if not matches:

        raise FileNotFoundError(
            f"No experiment folder found matching '{pattern}' "
            f"under experiments/. Check the model/k_shot/seed "
            f"combination exists, or that the dataset prefix is "
            f"correct."
        )

    # If multiple matches (re-runs), use the most recent
    exp_dir = matches[-1]

    history_path = exp_dir / "metrics" / "training_history.csv"

    if not history_path.exists():

        raise FileNotFoundError(
            f"{history_path} not found — experiment folder exists "
            f"but training_history.csv is missing."
        )

    return history_path, exp_dir.name


def main(dataset, k_shot, seed, prefix):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    used_folders = []

    for model in MODELS:

        history_path, folder_name = find_history(
            dataset, model, k_shot, seed
        )

        used_folders.append(folder_name)

        history = pd.read_csv(history_path)

        axes[0].plot(

            history["epoch"],

            history["accuracy"],

            label=MODEL_LABELS[model],

            color=PALETTE[model],

            linewidth=2,

        )

        axes[1].plot(

            history["epoch"],

            history["val_loss"],

            label=MODEL_LABELS[model],

            color=PALETTE[model],

            linewidth=2,

        )

    axes[0].set_xlabel("Epoch")

    axes[0].set_ylabel("Validation Accuracy")

    axes[0].set_title("Convergence: Validation Accuracy")

    axes[0].set_ylim(0, 1)

    axes[0].legend(loc="lower right", frameon=True)

    axes[1].set_xlabel("Epoch")

    axes[1].set_ylabel("Validation Loss")

    axes[1].set_title("Convergence: Validation Loss")

    axes[1].legend(loc="upper right", frameon=True)

    fig.suptitle(
        f"Training Convergence Comparison "
        f"({dataset}, {k_shot}-shot, seed {seed})",
        y=1.03,
    )

    out_path = FIG_DIR / f"{prefix}convergence_comparison.png"

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {out_path}")

    print("\nSource experiment folders used:")

    for f in used_folders:

        print(f"  {f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="CICIOT",
                         choices=["CICIOT", "TONIOT"])

    parser.add_argument("--k-shot", type=int, default=5)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prefix", default="")

    args = parser.parse_args()

    main(args.dataset, args.k_shot, args.seed, args.prefix)