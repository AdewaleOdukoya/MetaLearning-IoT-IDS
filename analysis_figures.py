"""
Dissertation-Level Summary Figures

Aggregate visualisations over a full sweep — these are the
figures for the results chapter, summarising all runs, as
opposed to the per-experiment plots in each experiment folder.

Reads:  the CSV given by --input
Writes: reports/figures/<prefix>*.png   (300 dpi)

Usage:
    python analysis_figures.py
    python analysis_figures.py --input reports/tables/TONIOT_sweep_raw_results.csv --prefix toniot_
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

FIG_DIR = Path("reports/figures")

FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

PREFIX = ""   # reassigned from CLI args in __main__

MODEL_ORDER = ["PROTONET", "MAML", "REPTILE", "FINETUNE"]

MODEL_LABELS = {
    "PROTONET": "Prototypical Networks",
    "MAML": "MAML (First-Order)",
    "REPTILE": "Reptile",
    "FINETUNE": "Fine-Tuned MLP (Transfer)",
}

PALETTE = {
    "PROTONET": "#1f77b4",
    "MAML": "#d62728",
    "REPTILE": "#2ca02c",
    "FINETUNE": "#9467bd",
}


def load(input_path):

    global MODEL_ORDER

    df = pd.read_csv(input_path)

    df = df[df["status"] == "OK"].copy()

    df["model_label"] = df["model"].map(MODEL_LABELS)

    # Keep only models actually present in this results file
    # (e.g. FINETUNE exists for CICIoT2023 but not TON-IoT)
    MODEL_ORDER = [
        m for m in MODEL_ORDER if m in df["model"].unique()
    ]

    return df


def save(fig, name):

    path = FIG_DIR / f"{PREFIX}{name}"

    fig.savefig(path, dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {path}")


# ================================================================
# Figure 1 — Accuracy vs K-shot, per model, seen vs zero-day
# ================================================================

def fig_accuracy_vs_kshot(df):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, split, title in [
        (axes[0], "seen", "Seen Classes"),
        (axes[1], "zero_day", "Zero-Day Classes"),
    ]:

        sub = df[df["split"] == split]

        for model in MODEL_ORDER:

            m = sub[sub["model"] == model]

            grouped = m.groupby("k_shot")["accuracy"]

            means = grouped.mean()

            stds = grouped.std()

            ax.errorbar(

                means.index,

                means.values,

                yerr=stds.values,

                label=MODEL_LABELS[model],

                color=PALETTE[model],

                marker="o",

                markersize=7,

                linewidth=2,

                capsize=4,

            )

        ax.set_title(title)

        ax.set_xlabel("K-Shot")

        ax.set_xticks(sorted(df["k_shot"].unique()))

        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Accuracy")

    axes[0].legend(loc="lower right", frameon=True)

    n_seeds = df["seed"].nunique()

    fig.suptitle(
        f"Few-Shot Accuracy vs Support-Set Size "
        f"(mean ± std, {n_seeds} seeds)",
        y=1.03,
    )

    save(fig, "accuracy_vs_kshot.png")


# ================================================================
# Figure 2 — Seen vs Zero-Day gap (grouped bars, at K=5)
# ================================================================

def fig_seen_vs_zeroday(df, k_shot=5):

    sub = df[df["k_shot"] == k_shot]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    sns.barplot(

        data=sub,

        x="model_label",

        y="accuracy",

        hue="split",

        order=[MODEL_LABELS[m] for m in MODEL_ORDER],

        hue_order=["seen", "zero_day"],

        errorbar="sd",

        capsize=0.08,

        palette={"seen": "#4c72b0", "zero_day": "#c44e52"},

        ax=ax,

    )

    ax.set_xlabel("")

    ax.set_ylabel("Accuracy")

    ax.set_ylim(0, 1)

    ax.set_title(
        f"Seen vs Zero-Day Accuracy ({k_shot}-shot, mean ± std)"
    )

    handles, _ = ax.get_legend_handles_labels()

    ax.legend(
        handles,
        ["Seen classes", "Zero-day classes"],
        frameon=True,
    )

    save(fig, f"seen_vs_zeroday_k{k_shot}.png")


# ================================================================
# Figure 3 — Zero-day degradation (accuracy drop from seen)
# ================================================================

def fig_zeroday_degradation(df):

    pivot = (

        df

        .groupby(["model", "k_shot", "split"])["accuracy"]

        .mean()

        .unstack("split")

    )

    pivot["drop"] = pivot["seen"] - pivot["zero_day"]

    pivot = pivot.reset_index()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    n_models = len(MODEL_ORDER)

    width = 0.8 / n_models

    k_shots = sorted(df["k_shot"].unique())

    x = np.arange(len(k_shots))

    offset_start = -(n_models - 1) / 2

    for i, model in enumerate(MODEL_ORDER):

        m = pivot[pivot["model"] == model].set_index("k_shot")

        drops = [m.loc[k, "drop"] for k in k_shots]

        ax.bar(

            x + (offset_start + i) * width,

            drops,

            width,

            label=MODEL_LABELS[model],

            color=PALETTE[model],

        )

    ax.set_xticks(x)

    ax.set_xticklabels([f"{k}-shot" for k in k_shots])

    ax.set_ylabel("Accuracy Drop (Seen − Zero-Day)")

    ax.set_title(
        "Generalisation Gap on Zero-Day Classes\n"
        "(lower = better zero-day generalisation)"
    )

    ax.axhline(0, color="black", linewidth=0.8)

    ax.legend(frameon=True)

    save(fig, "zeroday_degradation.png")


# ================================================================
# Figure 4 — Multi-metric comparison at K=5, zero-day split
# ================================================================

def fig_multimetric_zeroday(df, k_shot=5):

    metrics = ["accuracy", "f1_score", "roc_auc", "false_positive_rate"]

    metric_labels = {
        "accuracy": "Accuracy",
        "f1_score": "Macro F1",
        "roc_auc": "ROC-AUC",
        "false_positive_rate": "FPR",
    }

    sub = df[(df["k_shot"] == k_shot) & (df["split"] == "zero_day")]

    melted = sub.melt(

        id_vars=["model_label"],

        value_vars=metrics,

        var_name="metric",

        value_name="value",

    )

    melted["metric"] = melted["metric"].map(metric_labels)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    sns.barplot(

        data=melted,

        x="metric",

        y="value",

        hue="model_label",

        hue_order=[MODEL_LABELS[m] for m in MODEL_ORDER],

        errorbar="sd",

        capsize=0.06,

        palette=[PALETTE[m] for m in MODEL_ORDER],

        ax=ax,

    )

    ax.set_xlabel("")

    ax.set_ylabel("Score")

    ax.set_title(
        f"Zero-Day Detection Performance Across Metrics "
        f"({k_shot}-shot, mean ± std)\n"
        "(note: lower is better for FPR)"
    )

    ax.legend(title="", frameon=True)

    save(fig, f"multimetric_zeroday_k{k_shot}.png")


# ================================================================
# Figure 5 — Performance vs adaptation cost trade-off
# ================================================================

def fig_tradeoff(df, k_shot=5):

    # Prefer adaptation_time (backfilled); fall back to
    # inference_time if this results file predates the backfill.
    time_col = (
        "adaptation_time"
        if "adaptation_time" in df.columns
        and df["adaptation_time"].notna().any()
        else "inference_time"
    )

    time_label = (
        "Adaptation Time per Episode (s)"
        if time_col == "adaptation_time"
        else "Evaluation Time per Episode Batch (s)"
    )

    sub = df[(df["k_shot"] == k_shot) & (df["split"] == "zero_day")]

    agg = (

        sub

        .groupby("model")

        .agg(

            acc_mean=("accuracy", "mean"),

            acc_std=("accuracy", "std"),

            time_mean=(time_col, "mean"),

            time_std=(time_col, "std"),

        )

        .reset_index()

    )

    fig, ax = plt.subplots(figsize=(8.5, 6))

    for _, r in agg.iterrows():

        ax.errorbar(

            r["time_mean"],

            r["acc_mean"],

            xerr=r["time_std"],

            yerr=r["acc_std"],

            marker="o",

            markersize=12,

            capsize=4,

            color=PALETTE[r["model"]],

            label=MODEL_LABELS[r["model"]],

        )

        ax.annotate(

            MODEL_LABELS[r["model"]],

            (r["time_mean"], r["acc_mean"]),

            textcoords="offset points",

            xytext=(12, 8),

            fontsize=11,

        )

    ax.set_xlabel(time_label)

    ax.set_ylabel("Zero-Day Accuracy")

    ax.set_title(
        f"Detection Performance vs Computational Cost "
        f"({k_shot}-shot zero-day)"
    )

    save(fig, f"tradeoff_accuracy_vs_time_k{k_shot}.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="reports/tables/sweep_raw_results.csv",
    )

    parser.add_argument("--prefix", default="")

    args = parser.parse_args()

    PREFIX = args.prefix

    df = load(args.input)

    fig_accuracy_vs_kshot(df)

    fig_seen_vs_zeroday(df, k_shot=5)

    fig_zeroday_degradation(df)

    fig_multimetric_zeroday(df, k_shot=5)

    fig_tradeoff(df, k_shot=5)

    print(f"\nAll figures written to reports/figures/ "
          f"(prefix: '{PREFIX or 'none'}')")