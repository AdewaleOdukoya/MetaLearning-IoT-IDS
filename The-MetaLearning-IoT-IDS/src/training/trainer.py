"""
Generic Trainer

Supports:
- MLP
- CNN
- CNN-LSTM

Tracks:
- Train Loss
- Validation Loss
- Accuracy
- Precision
- Recall
- F1 Score

Automatically saves:
- Best model
- Training history
"""

from pathlib import Path
import pandas as pd
import torch

from src.evaluation.metrics import compute_metrics


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        experiment,
        epochs=2,
        checkpoint_name="model.pt",
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer

        self.device = device
        self.experiment = experiment
        self.epochs = epochs

        self.best_accuracy = 0.0

        # self.checkpoint_path = self.experiment.root / checkpoint_name
        self.checkpoint_path = (

            self.experiment.checkpoint_dir /

            checkpoint_name

        )

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": []
        }

    def train(self):

        self.model.to(self.device)

        print("=" * 80)
        print("Training Started")
        print("=" * 80)

        for epoch in range(1, self.epochs + 1):

            train_loss = self._train_one_epoch()

            val_loss, metrics = self._validate()

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["accuracy"].append(metrics["accuracy"])
            self.history["precision"].append(metrics["precision"])
            self.history["recall"].append(metrics["recall"])
            self.history["f1_score"].append(metrics["f1_score"])

            print(f"\nEpoch [{epoch}/{self.epochs}]")
            print("-" * 80)

            print(f"Train Loss      : {train_loss:.4f}")
            print(f"Validation Loss : {val_loss:.4f}")
            print(f"Accuracy        : {metrics['accuracy']:.4f}")
            print(f"Precision       : {metrics['precision']:.4f}")
            print(f"Recall          : {metrics['recall']:.4f}")
            print(f"F1 Score        : {metrics['f1_score']:.4f}")

            if metrics["accuracy"] > self.best_accuracy:

                self.best_accuracy = metrics["accuracy"]

                torch.save(
                    self.model.state_dict(),
                    self.checkpoint_path
                )

                print("✓ Best model updated.")

        history_df = pd.DataFrame(self.history)

        history_df.to_csv(
            self.experiment.metrics_dir / "training_history.csv",
            index=False,
        )

        print("\nTraining Finished.")
        print(f"Best Accuracy : {self.best_accuracy:.4f}")

        return self.history

    def _train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        for X, y in self.train_loader:

            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(X)

            loss = self.criterion(outputs, y)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def _validate(self):

        self.model.eval()

        running_loss = 0

        predictions = []

        labels = []

        with torch.no_grad():

            for X, y in self.val_loader:

                X = X.to(self.device)
                y = y.to(self.device)

                outputs = self.model(X)

                loss = self.criterion(outputs, y)

                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                predictions.extend(preds.cpu().numpy())

                labels.extend(y.cpu().numpy())

        metrics = compute_metrics(labels, predictions)

        return running_loss / len(self.val_loader), metrics