"""
Pretraining for the Transfer-Learning Fine-Tune Baseline

Trains a standard MLP (ordinary supervised learning) on the
meta-train classes ONLY — the same classes the meta-learners were
allowed to see, with ZERO_DAY_CLASSES excluded — so the comparison
is like-for-like.

Labels are remapped to a contiguous 0..C-1 range for CrossEntropy.

Saves:
  - Normal experiment folder (FINETUNE_PRETRAIN_<timestamp>)
  - A stable copy of the checkpoint + label mapping at
    results/pretrained/  (used by evaluate_finetune.py)

Usage:
    python -m src.training.pretrain_finetune
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.config import config as cfg
from src.data.dataset import CICIoTDataset
from src.models.mlp import MLP
from src.training.trainer import Trainer
from src.experiments.experiment_manager import ExperimentManager
from src.utils.seed import set_seed


class FilteredRemappedDataset(Dataset):
    """Restricts a dataset to allowed classes and remaps labels
    to a contiguous 0..C-1 range."""

    def __init__(self, base_dataset, allowed_classes):

        labels = base_dataset.y.numpy()

        allowed = np.array(sorted(allowed_classes))

        mask = np.isin(labels, allowed)

        self.indices = np.where(mask)[0]

        # original class id -> contiguous id
        self.label_map = {
            int(orig): new for new, orig in enumerate(allowed)
        }

        self.base = base_dataset

        remapped = np.array(
            [self.label_map[int(l)] for l in labels[self.indices]]
        )

        self.y = torch.tensor(remapped, dtype=torch.long)

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, i):

        x, _ = self.base[self.indices[i]]

        return x, self.y[i]


def main():

    set_seed(cfg.SEED)

    train_base = CICIoTDataset(
        feature_file="X_train.csv", label_file="y_train.csv")

    val_base = CICIoTDataset(
        feature_file="X_validation.csv", label_file="y_validation.csv")

    all_classes = np.unique(train_base.y.numpy())

    meta_train_classes = np.setdiff1d(
        all_classes, np.array(cfg.ZERO_DAY_CLASSES))

    train_ds = FilteredRemappedDataset(train_base, meta_train_classes)

    val_ds = FilteredRemappedDataset(val_base, meta_train_classes)

    num_classes = len(meta_train_classes)

    input_dim = train_base.X.shape[1]

    print(f"Pretraining on {num_classes} meta-train classes "
          f"({len(train_ds)} samples)")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    experiment = ExperimentManager("FINETUNE_PRETRAIN")

    experiment.create_readme()

    experiment.save_config({
        "model": "FINETUNE_PRETRAIN_MLP",
        "epochs": cfg.EPOCHS,
        "batch_size": cfg.BATCH_SIZE,
        "learning_rate": cfg.LEARNING_RATE,
        "num_classes": num_classes,
        "input_features": int(input_dim),
        "zero_day_classes_excluded": list(cfg.ZERO_DAY_CLASSES),
        "seed": cfg.SEED,
    })

    model = MLP(input_dim=input_dim, num_classes=num_classes)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE),
        device=cfg.DEVICE,
        experiment=experiment,
        epochs=cfg.EPOCHS,
        checkpoint_name="finetune_mlp.pt",
    )

    trainer.train()

    # -------- stable copy for the evaluation script --------

    stable_dir = Path("results/pretrained")

    stable_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        experiment.checkpoint_dir / "finetune_mlp.pt",
        stable_dir / "finetune_mlp.pt",
    )

    with open(stable_dir / "finetune_meta.json", "w") as f:

        json.dump({
            "num_classes": int(num_classes),
            "input_dim": int(input_dim),
            "meta_train_classes": [int(c) for c in meta_train_classes],
        }, f, indent=4)

    print(f"\nStable checkpoint saved to {stable_dir}/finetune_mlp.pt")


if __name__ == "__main__":

    main()