import numpy as np
from src.data.dataset import CICIoTDataset

train_dataset = CICIoTDataset(feature_file="X_train.csv", label_file="y_train.csv")
labels, counts = np.unique(train_dataset.y.numpy(), return_counts=True)
for label, count in sorted(zip(labels, counts), key=lambda x: x[1]):
    print(f"class {label}: {count} samples")