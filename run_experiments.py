"""
Experiment Sweep Runner  (Objective 4)

Runs the full model x k-shot x seed grid:

    {PROTONET, MAML, REPTILE} x {1, 5, 10}-shot x 5 seeds

For each run, records seen-class and zero-day metrics, then
aggregates everything into mean +/- std tables suitable for
direct inclusion in the dissertation.

Outputs:
    reports/tables/sweep_raw_results.csv
    reports/tables/sweep_summary.csv

Usage:
    python run_experiments.py
    python run_experiments.py --models MAML REPTILE --k-shots 1 5 --seeds 42 43
"""

import argparse
import time
import traceback
from pathlib import Path

import pandas as pd

from src.training.train_meta import main as train_meta_main


METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "balanced_accuracy",
    "false_positive_rate",
    "roc_auc",
    "loss",
    "inference_time",
]


def run_sweep(models, k_shots, seeds, epochs, episodes):

    results = []

    total = len(models) * len(k_shots) * len(seeds)

    run_index = 0

    for model_name in models:

        for k_shot in k_shots:

            for seed in seeds:

                run_index += 1

                print("\n" + "#" * 80)
                print(
                    f"# SWEEP RUN {run_index}/{total} — "
                    f"{model_name} | {k_shot}-shot | seed {seed}"
                )
                print("#" * 80)

                start = time.time()

                try:

                    output = train_meta_main(

                        model_name=model_name,

                        k_shot=k_shot,

                        seed=seed,

                        epochs=epochs,

                        episodes=episodes,

                        experiment_suffix=f"K{k_shot}_S{seed}",

                    )

                except Exception:

                    # One failed run must not kill a multi-hour sweep.
                    print(f"\n!!! RUN FAILED: {model_name} K={k_shot} seed={seed}")

                    traceback.print_exc()

                    results.append({

                        "model": model_name,

                        "k_shot": k_shot,

                        "seed": seed,

                        "status": "FAILED",

                    })

                    continue

                elapsed = time.time() - start

                for split_name, metrics in [
                    ("seen", output["seen_metrics"]),
                    ("zero_day", output["zero_day_metrics"]),
                ]:

                    if metrics is None:

                        continue

                    row = {

                        "model": model_name,

                        "k_shot": k_shot,

                        "seed": seed,

                        "split": split_name,

                        "experiment": output["experiment_name"],

                        "train_time_sec": elapsed,

                        "status": "OK",

                    }

                    for col in METRIC_COLUMNS:

                        if col in metrics:

                            row[col] = metrics[col]

                    results.append(row)

                # Save incrementally after every run, so a crash
                # or interruption never loses completed results.
                save_raw(results)

    return results


def save_raw(results):

    out_dir = Path("reports/tables")

    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(

        out_dir / "sweep_raw_results.csv",

        index=False,

    )


def aggregate(results):

    df = pd.DataFrame(results)

    df = df[df["status"] == "OK"]

    if df.empty:

        print("No successful runs to aggregate.")

        return

    agg_metrics = [
        c for c in METRIC_COLUMNS if c in df.columns
    ]

    summary = (

        df

        .groupby(["model", "k_shot", "split"])[agg_metrics]

        .agg(["mean", "std"])

        .round(4)

    )

    # Flatten column MultiIndex: accuracy_mean, accuracy_std, ...

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    out_path = Path("reports/tables") / "sweep_summary.csv"

    summary.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("SWEEP SUMMARY (mean ± std across seeds)")
    print("=" * 80)

    display_cols = [
        "model", "k_shot", "split",
        "accuracy_mean", "accuracy_std",
        "f1_score_mean", "f1_score_std",
    ]

    display_cols = [c for c in display_cols if c in summary.columns]

    print(summary[display_cols].to_string(index=False))

    print(f"\nFull summary saved to: {out_path}")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=["PROTONET", "MAML", "REPTILE"],
    )

    parser.add_argument(
        "--k-shots",
        nargs="+",
        type=int,
        default=[1, 5, 10],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
    )

    parser.add_argument("--epochs", type=int, default=None)

    parser.add_argument("--episodes", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    models = [m.upper() for m in args.models]

    results = run_sweep(

        models=models,

        k_shots=args.k_shots,

        seeds=args.seeds,

        epochs=args.epochs,

        episodes=args.episodes,

    )

    aggregate(results)