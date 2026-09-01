"""
Episode Generator for Few-Shot Meta-Learning

Supports:
- Prototypical Networks
- MAML
- Reptile
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    """
    Generates N-way K-shot episodes for meta-learning.

    Pass `allowed_classes` to restrict which classes this episode
    generator is permitted to sample from — this is what makes a
    disjoint meta-train / zero-day class split possible. When None,
    all classes present in the underlying dataset are used (original
    behaviour, unchanged).
    """

    def __init__(
        self,
        dataset,
        n_way=5,
        k_shot=5,
        query_size=15,
        episodes=1000,
        allowed_classes=None,
    ):

        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.episodes = episodes

        # Labels from the underlying dataset
        self.labels = dataset.y.numpy()

        # All unique classes present in this dataset
        all_classes = np.unique(self.labels)

        if allowed_classes is not None:

            allowed_classes = np.array(list(allowed_classes))

            self.classes = np.intersect1d(all_classes, allowed_classes)

            if len(self.classes) == 0:

                raise ValueError(

                    "None of the allowed_classes were found in this "
                    "dataset's labels. Check that allowed_classes "
                    "matches the dataset's label encoding."

                )

        else:

            self.classes = all_classes

        if self.n_way > len(self.classes):

            raise ValueError(

                f"n_way={self.n_way} exceeds the number of available "
                f"classes ({len(self.classes)}) for this episode "
                f"generator. Reduce n_way or provide more classes."

            )

        # Dictionary:
        # class -> indices of samples belonging to that class
        self.class_to_indices = {
            cls: np.where(self.labels == cls)[0]
            for cls in self.classes
        }

    def __len__(self):
        return self.episodes

    def __getitem__(self, index):

        support_x = []
        support_y = []

        query_x = []
        query_y = []

        # Randomly select N classes
        selected_classes = np.random.choice(
            self.classes,
            self.n_way,
            replace=False
        )

        for local_label, cls in enumerate(selected_classes):

            indices = self.class_to_indices[cls]

            # Select support + query samples
            chosen_indices = np.random.choice(
                indices,
                self.k_shot + self.query_size,
                replace=False
            )

            support_indices = chosen_indices[:self.k_shot]
            query_indices = chosen_indices[self.k_shot:]

            # Support Set
            for idx in support_indices:
                x, _ = self.dataset[idx]
                support_x.append(x)
                support_y.append(local_label)

            # Query Set
            for idx in query_indices:
                x, _ = self.dataset[idx]
                query_x.append(x)
                query_y.append(local_label)

        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y, dtype=torch.long)

        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y, dtype=torch.long)

        return support_x, support_y, query_x, query_y