"""
Backfill Adaptation Times for Existing Sweep Runs

For each completed sweep experiment (ProtoNet / MAML / Reptile),
loads the saved checkpoint and measures per-episode adaptation
time on both the seen-class and zero-day episode distributions.

IMPORTANT: this ONLY fills the `adaptation_time` column in
sweep_raw_results.csv. All existing metric values (accuracy, F1,
etc.) are left completely untouched — no re-evaluation of
performance, no overwriting of reported results.

A timestamped backup of sweep_raw_results.csv is made first.

Prerequisites: the adaptation_time patches must be applied to
maml_algorithm.py, reptile_algorithm.py, protonet_algorithm.py.

Usage:
    python backfill_adaptation_time.py
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import config as cfg
from src.data.dataset import CICIoTDataset
from src.data.episodes import EpisodeDataset
from src.data.meta_dataloader import get_zero_day_dataloader
from src.models.meta_factory import get_meta_model
from src.meta.algorithms.protonet_algorithm import ProtoNetAlgorithm
from src.meta.algorithms.maml_algorithm import MAMLAlgorithm
from src.meta.algorithms.reptile_algorithm import ReptileAlgorithm
from src.utils.seed import set_seed


# Episodes used purely for timing measurement. 50 gives a stable
# mean; timing does not need the full evaluation episode count.
TIMING_EPISODES = 50

RAW_PATH = Path("reports/tables/sweep_raw_results.csv")


def build_algorithm(model_name, model):

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


def measure(model, algorithm, loader, device):
    """Runs episodes through validation_step, returns mean
    adaptation_time. Performance outputs are discarded."""

    times = []

    for episode in loader:

        support_x, support_y, query_x, query_y = episode

        support_x = support_x.squeeze(0).to(device)
        support_y = support_y.squeeze(0).to(device)
        query_x = query_x.squeeze(0).to(device)
        query_y = query_y.squeeze(0).to(device)

        result = algorithm.validation_step(
            model, support_x, support_y, query_x, query_y)

        times.append(result.get("adaptation_time", 0.0))

    return float(np.mean(times))


def main():

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"{RAW_PATH} not found.")

    # ------- backup first -------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = RAW_PATH.with_name(
        f"{RAW_PATH.stem}_backup_{stamp}{RAW_PATH.suffix}")
    shutil.copy2(RAW_PATH, backup_path)
    print(f"Backed up raw results -> {backup_path.name}")

    raw = pd.read_csv(RAW_PATH)

    if "adaptation_time" not in raw.columns:
        raw["adaptation_time"] = np.nan

    # ------- datasets -------
    val_dataset = CICIoTDataset(
        feature_file="X_validation.csv", label_file="y_validation.csv")

    test_dataset = CICIoTDataset(
        feature_file="X_test.csv", label_file="y_test.csv")

    input_dim = val_dataset.X.shape[1]

    all_classes = np.unique(val_dataset.y.numpy())

    meta_train_classes = np.setdiff1d(
        all_classes, np.array(cfg.ZERO_DAY_CLASSES))

    # ------- walk experiment folders -------
    for exp_dir in sorted(Path("experiments").iterdir()):

        config_path = exp_dir / "config" / "config.json"

        if not config_path.exists():
            continue

        with open(config_path) as f:
            run_config = json.load(f)

        model_name = run_config.get("model")

        if model_name not in ("PROTONET", "MAML", "REPTILE"):
            continue

        if "seed" not in run_config:
            continue

        k_shot = run_config["k_shot"]
        seed = run_config["seed"]

        checkpoint = exp_dir / "checkpoints" / f"{model_name.lower()}.pt"

        if not checkpoint.exists():
            print(f"Skipping (no checkpoint): {exp_dir.name}")
            continue

        # Skip if both rows already have adaptation_time filled
        mask_base = (
            (raw["model"] == model_name) &
            (raw["k_shot"] == k_shot) &
            (raw["seed"] == seed)
        )

        if raw.loc[mask_base, "adaptation_time"].notna().all() \
                and mask_base.any():
            print(f"Already filled: {exp_dir.name}")
            continue

        print("\n" + "=" * 70)
        print(f"Timing: {exp_dir.name} "
              f"({model_name} | K={k_shot} | seed={seed})")
        print("=" * 70)

        set_seed(seed)

        model = get_meta_model(model_name, input_dim)

        model.load_state_dict(
            torch.load(checkpoint, map_location=cfg.DEVICE))

        model.to(cfg.DEVICE)

        algorithm = build_algorithm(model_name, model)

        # seen-class timing episodes
        seen_loader = DataLoader(
            EpisodeDataset(
                dataset=val_dataset,
                n_way=cfg.N_WAY,
                k_shot=k_shot,
                query_size=cfg.QUERY_SIZE,
                episodes=TIMING_EPISODES,
                allowed_classes=meta_train_classes,
            ),
            batch_size=1, shuffle=False,
        )

        zero_day_loader = get_zero_day_dataloader(
            test_dataset=test_dataset,
            n_way=cfg.N_WAY,
            k_shot=k_shot,
            query_size=cfg.QUERY_SIZE,
            episodes=TIMING_EPISODES,
            zero_day_classes=cfg.ZERO_DAY_CLASSES,
        )

        for split, loader in [
            ("seen", seen_loader),
            ("zero_day", zero_day_loader),
        ]:

            mean_time = measure(
                model, algorithm, loader, cfg.DEVICE)

            mask = mask_base & (raw["split"] == split)

            raw.loc[mask, "adaptation_time"] = mean_time

            print(f"  {split:>8}: {mean_time*1000:.2f} ms/episode "
                  f"({int(mask.sum())} row(s) updated)")

        # incremental save after each experiment
        raw.to_csv(RAW_PATH, index=False)

    # ------- regenerate summary -------
    ok = raw[raw["status"] == "OK"]

    metric_cols = [
        "accuracy", "precision", "recall", "f1_score",
        "balanced_accuracy", "false_positive_rate",
        "roc_auc", "loss", "inference_time", "adaptation_time",
    ]

    agg = [c for c in metric_cols if c in ok.columns]

    summary = (
        ok.groupby(["model", "k_shot", "split"])[agg]
        .agg(["mean", "std"]).round(6)
    )

    summary.columns = [f"{m}_{s}" for m, s in summary.columns]

    summary.reset_index().to_csv(
        "reports/tables/sweep_summary.csv", index=False)

    print("\nBackfill complete. Summary regenerated with "
          "adaptation_time columns.")


if __name__ == "__main__":

    main()