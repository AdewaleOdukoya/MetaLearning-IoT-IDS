"""
PyTorch Dataset for processed IDS datasets
(CICIoT2023 by default; TON-IoT via data_dir="toniot")
"""

from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED = PROJECT_ROOT / "data" / "processed"


class CICIoTDataset(Dataset):

    def __init__(self, feature_file, label_file, data_dir=None):

        base = PROCESSED / data_dir if data_dir else PROCESSED

        feature_path = base / feature_file
        label_path = base / label_file

        if not feature_path.exists():
            raise FileNotFoundError(feature_path)

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        self.X = pd.read_csv(feature_path).values
        self.y = pd.read_csv(label_path).values.squeeze()

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        return self.X[index], self.y[index]