"""
PyTorch DataLoaders
"""

from torch.utils.data import DataLoader

from src.data.dataset import CICIoTDataset


def get_dataloaders(batch_size=512, num_workers=4):

    train_dataset = CICIoTDataset(
        "X_train.csv",
        "y_train.csv"
    )

    validation_dataset = CICIoTDataset(
        "X_validation.csv",
        "y_validation.csv"
    )

    test_dataset = CICIoTDataset(
        "X_test.csv",
        "y_test.csv"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    return train_loader, validation_loader, test_loader