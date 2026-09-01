"""
Paired Statistical Tests over Sweep Results

For every (k_shot, split) condition, compares each pair of models
on accuracy and macro-F1 using:

  - Paired t-test (pairing by seed)
  - Wilcoxon signed-rank test (nonparametric companion,
    robust to non-normality with small n)

Outputs:
    reports/tables/<prefix>statistical_tests.csv

Usage:
    python analysis_ttests.py
    python analysis_ttests.py --input reports/tables/TONIOT_sweep_raw_results.csv --prefix toniot_
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS_TO_TEST = ["accuracy", "f1_score"]

ALPHA = 0.05


def main(input_path, prefix):

    df = pd.read_csv(input_path)

    df = df[df["status"] == "OK"]

    models = sorted(df["model"].unique())

    k_shots = sorted(df["k_shot"].unique())

    splits = sorted(df["split"].unique())

    rows = []

    for split in splits:

        for k_shot in k_shots:

            subset = df[
                (df["split"] == split) &
                (df["k_shot"] == k_shot)
            ]

            for model_a, model_b in combinations(models, 2):

                a = (
                    subset[subset["model"] == model_a]
                    .sort_values("seed")
                )

                b = (
                    subset[subset["model"] == model_b]
                    .sort_values("seed")
                )

                # Pair strictly on seed
                common_seeds = np.intersect1d(
                    a["seed"].values,
                    b["seed"].values,
                )

                a = a[a["seed"].isin(common_seeds)]

                b = b[b["seed"].isin(common_seeds)]

                if len(common_seeds) < 3:

                    continue

                for metric in METRICS_TO_TEST:

                    x = a[metric].values

                    y = b[metric].values

                    diff = x - y

                    t_stat, t_p = stats.ttest_rel(x, y)

                    # Wilcoxon requires non-identical samples
                    try:

                        w_stat, w_p = stats.wilcoxon(x, y)

                    except ValueError:

                        w_stat, w_p = np.nan, np.nan

                    rows.append({

                        "split": split,

                        "k_shot": k_shot,

                        "metric": metric,

                        "model_a": model_a,

                        "model_b": model_b,

                        "n_seeds": len(common_seeds),

                        f"mean_{metric}_a": round(x.mean(), 4),

                        f"mean_{metric}_b": round(y.mean(), 4),

                        "mean_diff_a_minus_b": round(diff.mean(), 4),

                        "t_statistic": round(t_stat, 4),

                        "t_p_value": round(t_p, 5),

                        "t_significant": t_p < ALPHA,

                        "wilcoxon_p_value": (
                            round(w_p, 5)
                            if not np.isnan(w_p) else None
                        ),

                        "better_model": (
                            model_a if diff.mean() > 0 else model_b
                        ),

                    })

    out = pd.DataFrame(rows)

    out_path = Path(f"reports/tables/{prefix}statistical_tests.csv")

    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}\n")

    # Readable console summary for accuracy
    acc = out[out["metric"] == "accuracy"]

    print("=" * 90)
    print("PAIRED T-TESTS — ACCURACY (pairing by seed, alpha=0.05)")
    print("=" * 90)

    for _, r in acc.iterrows():

        sig = "SIGNIFICANT" if r["t_significant"] else "not significant"

        print(
            f"[{r['split']:>8} | {r['k_shot']:>2}-shot] "
            f"{r['model_a']} vs {r['model_b']}: "
            f"diff={r['mean_diff_a_minus_b']:+.4f} "
            f"(better: {r['better_model']}), "
            f"p={r['t_p_value']:.5f} -> {sig}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="reports/tables/sweep_raw_results.csv",
    )

    parser.add_argument("--prefix", default="")

    args = parser.parse_args()

    main(input_path=args.input, prefix=args.prefix)