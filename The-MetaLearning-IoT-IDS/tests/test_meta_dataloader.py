"""
Test Meta DataLoader
"""

from src.data.dataset import CICIoTDataset
from src.data.meta_dataloader import get_episode_dataloaders

# -----------------------------------------------------
# Load Processed Dataset
# -----------------------------------------------------

dataset = CICIoTDataset(

    feature_file="X_train.csv",

    label_file="y_train.csv",

)

# -----------------------------------------------------
# Create Episode Loaders
# -----------------------------------------------------

train_loader, val_loader = get_episode_dataloaders(

    train_dataset=dataset,

    val_dataset=dataset,

    n_way=5,

    k_shot=5,

    query_size=10,

    episodes=100,

)

# -----------------------------------------------------
# Inspect One Episode
# -----------------------------------------------------

episode = next(iter(train_loader))

support_x, support_y, query_x, query_y = episode

print()

print("=" * 60)
print("Episode Shapes")
print("=" * 60)

print("Support X :", support_x.shape)
print("Support y :", support_y.shape)

print()

print("Query X   :", query_x.shape)
print("Query y   :", query_y.shape)