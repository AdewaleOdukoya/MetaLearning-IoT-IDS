"""
Generic Meta-Learning Trainer

Supports:

- Prototypical Networks
- MAML
- Reptile

Responsibilities

- Episode Training
- Validation
- Logging
- Checkpointing
- Experiment Management

The algorithm-specific optimisation is delegated to
the model implementation.
"""

from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


class MetaTrainer:

    def __init__(
        self,
        model,
        algorithm,
        train_episodes,
        val_episodes,
        optimizer,
        device,
        experiment,
        epochs=20,
        checkpoint_name="meta_model.pt",
    ):

        self.model = model

        self.algorithm = algorithm

        self.train_episodes = train_episodes

        self.val_episodes = val_episodes

        self.optimizer = optimizer

        self.device = device

        self.experiment = experiment

        self.epochs = epochs

        self.checkpoint_path = (
            self.experiment.checkpoint_dir /
            checkpoint_name
        )

        self.best_accuracy = 0

        self.history = {

            "epoch": [],

            "train_loss": [],

            "val_loss": [],

            "accuracy": [],

            "precision": [],

            "recall": [],

            "f1_score": [],

        }

        # ==========================================================
    # Train Meta-Learning Model
    # ==========================================================

    def train(self):

        self.model.to(self.device)

        print("\n" + "=" * 80)
        print("META-LEARNING TRAINING")
        print("=" * 80)

        for epoch in range(1, self.epochs + 1):

            train_loss, train_accuracy = self.train_epoch()

            val_loss, val_accuracy, val_precision, val_recall, val_f1 = self.validate()

            # -------------------------------------------------
            # Save History
            # -------------------------------------------------

            self.history["epoch"].append(epoch)

            self.history["train_loss"].append(train_loss)

            self.history["val_loss"].append(val_loss)

            self.history["accuracy"].append(val_accuracy)

            self.history["precision"].append(val_precision)

            self.history["recall"].append(val_recall)

            self.history["f1_score"].append(val_f1)

            # -------------------------------------------------
            # Save Best Model
            # -------------------------------------------------

            if val_accuracy > self.best_accuracy:

                self.best_accuracy = val_accuracy

                torch.save(

                    self.model.state_dict(),

                    self.checkpoint_path,

                )

            # -------------------------------------------------
            # Print Progress
            # -------------------------------------------------

            print(

                f"Epoch [{epoch}/{self.epochs}] "

                f"| Train Loss: {train_loss:.4f} "

                f"| Train Acc: {train_accuracy:.4f} "

                f"| Val Loss: {val_loss:.4f} "

                f"| Val Acc: {val_accuracy:.4f}"

            )

        # -----------------------------------------------------
        # Save Training History
        # -----------------------------------------------------

        history_df = pd.DataFrame(

            self.history

        )

        history_df.to_csv(

            self.experiment.metrics_dir /

            "training_history.csv",

            index=False,

        )

        print("\nTraining Finished.")

        print(

            f"Best Validation Accuracy: "

            f"{self.best_accuracy:.4f}"

        )

        return self.history
    
       # ==========================================================
    # Train One Epoch
    # ==========================================================

    def train_epoch(self):

        self.model.train()

        running_loss = 0.0

        running_accuracy = 0.0

        total_episodes = len(self.train_episodes)

        for episode in self.train_episodes:

            (
                support_x,
                support_y,
                query_x,
                query_y,
            ) = episode

            # ---------------------------------------------
            # Remove Episode Dimension
            # ---------------------------------------------

            support_x = support_x.squeeze(0).to(self.device)

            support_y = support_y.squeeze(0).to(self.device)

            query_x = query_x.squeeze(0).to(self.device)

            query_y = query_y.squeeze(0).to(self.device)

            # ---------------------------------------------
            # Forward
            # ---------------------------------------------

            result = self.algorithm.training_step(

                self.model,

                support_x,

                support_y,

                query_x,

                query_y,

            )

            loss = result["loss"]

            accuracy = result["accuracy"]

            # ---------------------------------------------
            # Backpropagation
            # ---------------------------------------------

            if not result.get("skip_optimizer_step", False):

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

            # ---------------------------------------------
            # Statistics
            # ---------------------------------------------

            running_loss += loss.item()

            running_accuracy += accuracy.item()

        epoch_loss = running_loss / total_episodes

        epoch_accuracy = running_accuracy / total_episodes

        return epoch_loss, epoch_accuracy


        # ==========================================================
    # Validate
    # ==========================================================

    def validate(self):

        self.model.eval()

        running_loss = 0.0

        running_accuracy = 0.0

        total_episodes = len(self.val_episodes)

        all_y_true = []

        all_y_pred = []

        for episode in self.val_episodes:

            (
                support_x,
                support_y,
                query_x,
                query_y,
            ) = episode

            # ---------------------------------------------
            # Remove Episode Dimension
            # ---------------------------------------------

            support_x = support_x.squeeze(0).to(self.device)

            support_y = support_y.squeeze(0).to(self.device)

            query_x = query_x.squeeze(0).to(self.device)

            query_y = query_y.squeeze(0).to(self.device)

            # ---------------------------------------------
            # Validation
            # ---------------------------------------------

            result = self.algorithm.validation_step(

                self.model,

                support_x,

                support_y,

                query_x,

                query_y,

            )

            running_loss += result["loss"].item()

            running_accuracy += result["accuracy"].item()

            all_y_true.extend(query_y.detach().cpu().numpy())

            all_y_pred.extend(result["predictions"].detach().cpu().numpy())

        epoch_loss = running_loss / total_episodes

        epoch_accuracy = running_accuracy / total_episodes

        # ---------------------------------------------
        # Macro Precision / Recall / F1 across all
        # validation episodes this epoch
        # ---------------------------------------------

        precision, recall, f1, _ = precision_recall_fscore_support(

            np.array(all_y_true),

            np.array(all_y_pred),

            average="macro",

            zero_division=0,

        )

        return epoch_loss, epoch_accuracy, precision, recall, f1