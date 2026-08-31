from src.data.dataset import CICIoTDataset
from src.data.episodes import EpisodeDataset

from src.config.config import (
    N_WAY,
    K_SHOT,
    QUERY_SIZE,
    EPISODES,
)

dataset = CICIoTDataset(
    "X_train.csv",
    "y_train.csv"
)

episode_dataset = EpisodeDataset(
    dataset=dataset,
    n_way=N_WAY,
    k_shot=K_SHOT,
    query_size=QUERY_SIZE,
    episodes=EPISODES,
)

support_x, support_y, query_x, query_y = episode_dataset[0]

print("Support X:", support_x.shape)
print("Support y:", support_y.shape)

print("Query X:", query_x.shape)
print("Query y:", query_y.shape)