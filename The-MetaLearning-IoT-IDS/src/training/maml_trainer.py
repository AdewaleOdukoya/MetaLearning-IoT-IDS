"""
MAML Trainer

Dedicated trainer for optimisation-based meta-learning
using First-Order MAML (FOMAML).
"""

import higher
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd


class MAMLTrainer:

    def __init__(
        self,
        model,
        train_episodes,
        val_episodes,
        device,
        experiment,
        meta_lr,
        inner_lr,
        inner_steps,
        epochs,
        checkpoint_name="maml.pt",
    ):

        # --------------------------------------------------
        # Core Components
        # --------------------------------------------------

        self.model = model

        self.train_episodes = train_episodes

        self.val_episodes = val_episodes

        self.device = device

        self.experiment = experiment

        # --------------------------------------------------
        # Hyperparameters
        # --------------------------------------------------

        self.meta_lr = meta_lr

        self.inner_lr = inner_lr

        self.inner_steps = inner_steps

        self.epochs = epochs

        # --------------------------------------------------
        # Meta Optimizer
        # --------------------------------------------------

        self.meta_optimizer = optim.Adam(

            self.model.parameters(),

            lr=self.meta_lr,

        )

        # --------------------------------------------------
        # Checkpoint
        # --------------------------------------------------

        self.checkpoint_path = (

            self.experiment.checkpoint_dir /

            checkpoint_name

        )

        # --------------------------------------------------
        # Best Validation Accuracy
        # --------------------------------------------------

        self.best_accuracy = 0.0

        # --------------------------------------------------
        # Training History
        # --------------------------------------------------

        self.history = {

            "epoch": [],

            "train_loss": [],

            "train_accuracy": [],

            "val_loss": [],

            "val_accuracy": [],

        }

        # --------------------------------------------------
        # Move Model to Device
        # --------------------------------------------------

        self.model.to(self.device)

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

            # --------------------------------------------------
            # Remove Batch Dimension
            # --------------------------------------------------

            support_x = support_x.squeeze(0).to(self.device)
            support_y = support_y.squeeze(0).to(self.device)

            query_x = query_x.squeeze(0).to(self.device)
            query_y = query_y.squeeze(0).to(self.device)

            # --------------------------------------------------
            # Create Inner Optimizer
            # --------------------------------------------------

            inner_optimizer = torch.optim.SGD(

                self.model.parameters(),

                lr=self.inner_lr,

            )

            # --------------------------------------------------
            # Reset Meta Gradients
            # --------------------------------------------------

            self.meta_optimizer.zero_grad()

            # --------------------------------------------------
            # Differentiable Inner Loop
            # --------------------------------------------------

            with higher.innerloop_ctx(

                self.model,

                inner_optimizer,

                copy_initial_weights=False,

                track_higher_grads=False,

            ) as (fmodel, diffopt):

                # ----------------------------------------------
                # Inner Adaptation
                # ----------------------------------------------

                for _ in range(self.inner_steps):

                    support_logits = fmodel(

                        support_x

                    )

                    support_loss = F.cross_entropy(

                        support_logits,

                        support_y,

                    )

                    diffopt.step(

                        support_loss

                    )

                # ----------------------------------------------
                # Query Prediction
                # ----------------------------------------------

                query_logits = fmodel(

                    query_x

                )

                query_loss = F.cross_entropy(

                    query_logits,

                    query_y,

                )

                predictions = torch.argmax(

                    query_logits,

                    dim=1,

                )

                accuracy = (

                    predictions == query_y

                ).float().mean()

            # --------------------------------------------------
            # Meta Update
            # --------------------------------------------------

            query_loss.backward()

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                max_norm=5.0,

            )

            self.meta_optimizer.step()

            # --------------------------------------------------
            # Statistics
            # --------------------------------------------------

            running_loss += query_loss.item()

            running_accuracy += accuracy.item()

        epoch_loss = running_loss / total_episodes

        epoch_accuracy = running_accuracy / total_episodes

        return epoch_loss, epoch_accuracy