"""
Training Script for Baseline Deep Learning Models

Current Model
-------------
- MLP

Future Models
-------------
- CNN
- CNN-LSTM

This script:
1. Loads the processed dataset
2. Builds the model
3. Trains the model
4. Saves the best checkpoint
5. Saves training history
6. Generates training visualisations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.experiments.experiment_manager import ExperimentManager
from src.evaluation.evaluate import Evaluator
import os
torch.set_num_threads(os.cpu_count())

from src.config.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DEVICE,
    MODEL_NAME,
)

from src.data.dataloader import get_dataloaders

from src.models.baseline_factory import get_model

from src.training.trainer import Trainer

from src.visualization.visualizer import Visualizer



def main():

    print("=" * 80)
    print("Loading Dataset...")
    print("=" * 80)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    # -------------------------------------------------------------
    # Automatically infer input dimensions
    # -------------------------------------------------------------

    sample_features, _ = next(iter(train_loader))

    input_dim = sample_features.shape[1]

    num_classes = len(
        np.unique(
            train_loader.dataset.y.numpy()
        )
    )

    print(f"Input Features   : {input_dim}")
    print(f"Number of Classes: {num_classes}")

    # -------------------------------------------------------------
    # Experiment Manager
    # -------------------------------------------------------------

    experiment = ExperimentManager(MODEL_NAME)
    experiment.create_readme()

    # -------------------------------------------------------------
    # Build Model
    # -------------------------------------------------------------

    model = get_model(
        model_name=MODEL_NAME,
        input_dim=input_dim,
        num_classes=num_classes,
    )

    print("\nModel Architecture\n")
    print(model)

    experiment.save_config({

        "model": MODEL_NAME,

        "epochs": EPOCHS,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "optimizer": "Adam",

        "loss": "CrossEntropyLoss",

        "input_features": input_dim,

        "num_classes": num_classes,

        "device": DEVICE,

    })

    # -------------------------------------------------------------
    # Loss Function
    # -------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        experiment=experiment,
        epochs=EPOCHS,
        checkpoint_name=f"best_{MODEL_NAME.lower()}.pt",
    )

    # -------------------------------------------------------------
    # Train
    # -------------------------------------------------------------

    history = trainer.train()

    # -------------------------------------------------------------
    # Generate Training Figures
    # -------------------------------------------------------------

    print("\nGenerating Training Visualisations...")

    visualizer = Visualizer(
        experiment.plots_dir
    )

    visualizer.plot_loss(history)

    visualizer.plot_accuracy(history)

    visualizer.plot_metrics(history)

    # -------------------------------------------------------------
    # Copy Training Plots
    # -------------------------------------------------------------

    # experiment.copy_plots()

    # -------------------------------------------------------------
    # Evaluate Best Model
    # -------------------------------------------------------------

    print("\nStarting Test Evaluation...")

    evaluator = Evaluator(

        model=model,

        test_loader=test_loader,

        device=DEVICE,

        checkpoint_path=experiment.checkpoint_dir / f"best_{MODEL_NAME.lower()}.pt",

        experiment=experiment,

        model_name=MODEL_NAME,

    )

    evaluator.evaluate()

    print("\nTraining pipeline completed successfully.")

    # experiment.summary()


if __name__ == "__main__":
    main()