import numpy as np
from src.data.dataset import CICIoTDataset
from src.config.config import ZERO_DAY_CLASSES, QUERY_SIZE

test_dataset = CICIoTDataset(feature_file="X_test.csv", label_file="y_test.csv")
labels, counts = np.unique(test_dataset.y.numpy(), return_counts=True)
count_map = dict(zip(labels, counts))

needed = 10 + QUERY_SIZE  # worst case in your sweep: 10-shot
print(f"Each zero-day class needs >= {needed} test samples\n")

for cls in ZERO_DAY_CLASSES:
    n = count_map.get(cls, 0)
    status = "OK" if n >= needed else "TOO FEW  <-- problem"
    print(f"class {cls}: {n} samples  [{status}]")