"""
Meta DataLoader

Creates episodic DataLoaders for
few-shot meta-learning.

Supports:

- ProtoNet
- MAML
- Reptile
"""

import numpy as np
from torch.utils.data import DataLoader

from src.data.episodes import EpisodeDataset


def get_episode_dataloaders(

    train_dataset,

    val_dataset,

    n_way,

    k_shot,

    query_size,

    episodes,

    zero_day_classes=None,

):
    """
    Builds meta-training and meta-validation episode loaders.

    If `zero_day_classes` is provided, those classes are excluded
    from both the training and validation episode pools — they are
    reserved entirely for a separate zero-day evaluation loader
    (see get_zero_day_dataloader below). This is what creates a
    genuine held-out-class split rather than ordinary few-shot
    learning over all seen classes.

    Returns train_loader, val_loader, and the array of class labels
    that were actually used for meta-training (useful for logging /
    experiment tracking, so you can record exactly which classes
    were "seen" for a given run).
    """

    all_classes = np.unique(train_dataset.y.numpy())

    if zero_day_classes is not None and len(zero_day_classes) > 0:

        zero_day_classes = np.array(list(zero_day_classes))

        meta_train_classes = np.setdiff1d(all_classes, zero_day_classes)

    else:

        meta_train_classes = all_classes

    train_episode_dataset = EpisodeDataset(

        dataset=train_dataset,

        n_way=n_way,

        k_shot=k_shot,

        query_size=query_size,

        episodes=episodes,

        allowed_classes=meta_train_classes,

    )

    val_episode_dataset = EpisodeDataset(

        dataset=val_dataset,

        n_way=n_way,

        k_shot=k_shot,

        query_size=query_size,

        episodes=max(
            episodes // 5,
            100,
        ),

        allowed_classes=meta_train_classes,

    )

    train_loader = DataLoader(

        train_episode_dataset,

        batch_size=1,

        shuffle=True,

    )

    val_loader = DataLoader(

        val_episode_dataset,

        batch_size=1,

        shuffle=False,

    )

    return (

        train_loader,

        val_loader,

        meta_train_classes,

    )


def get_zero_day_dataloader(

    test_dataset,

    n_way,

    k_shot,

    query_size,

    episodes,

    zero_day_classes,

):
    """
    Builds an episode loader that ONLY samples from the held-out
    zero-day classes, drawn from a dataset split the model has
    never touched during training or validation (i.e. your true
    test set). This is the loader you run the final zero-day
    evaluation on, after training is completely finished.
    """

    if zero_day_classes is None or len(zero_day_classes) == 0:

        raise ValueError(

            "zero_day_classes must be a non-empty list of class "
            "labels to build a zero-day evaluation loader."

        )

    zero_day_episode_dataset = EpisodeDataset(

        dataset=test_dataset,

        n_way=n_way,

        k_shot=k_shot,

        query_size=query_size,

        episodes=episodes,

        allowed_classes=np.array(list(zero_day_classes)),

    )

    zero_day_loader = DataLoader(

        zero_day_episode_dataset,

        batch_size=1,

        shuffle=False,

    )

    return zero_day_loader