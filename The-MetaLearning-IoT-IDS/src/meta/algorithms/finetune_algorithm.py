"""
Transfer-Learning Fine-Tune Baseline (non-meta baseline)

The conventional alternative to meta-learning:

  1. A backbone MLP is PRETRAINED with ordinary supervised learning
     on the meta-train classes (done once, separately — see
     src/training/pretrain_finetune.py).
  2. At evaluation time, for each N-way K-shot episode:
       a. Replace the output head with a fresh N-way linear layer.
       b. Fine-tune the whole network on the K*N support samples.
       c. Evaluate on the query set.
       d. Restore the original pretrained weights (episodes must
          not contaminate each other).

Conforms to the same validation_step contract as the meta
algorithms ({loss, accuracy, logits, predictions, adaptation_time}),
so it plugs directly into MetaEvaluator and the existing episode
loaders — meaning it is evaluated on the EXACT same episodes,
splits, and seeds as ProtoNet/MAML/Reptile.

No training_step: this baseline is never meta-trained.
"""

import copy
import time

import torch
import torch.nn as nn


class FinetuneBaselineAlgorithm:

    def __init__(
        self,
        finetune_lr=0.01,
        finetune_steps=20,
        device="cpu",
        head_layer_index=6,   # index of the final Linear inside MLP.network
    ):

        self.finetune_lr = finetune_lr

        self.finetune_steps = finetune_steps

        self.device = device

        self.head_layer_index = head_layer_index

        self.criterion = nn.CrossEntropyLoss()

    def validation_step(self, model, support_x, support_y, query_x, query_y):

        # ------------------------------------------------
        # Snapshot pretrained weights + original head shape
        # ------------------------------------------------

        original_state = copy.deepcopy(model.state_dict())

        old_head = model.network[self.head_layer_index]

        in_features = old_head.in_features

        original_out = old_head.out_features

        n_way = int(support_y.max().item()) + 1

        # ------------------------------------------------
        # Fresh N-way head for this episode
        # ------------------------------------------------

        model.network[self.head_layer_index] = nn.Linear(
            in_features, n_way
        ).to(self.device)

        # ------------------------------------------------
        # Fine-tune on the support set  (timed = adaptation)
        # ------------------------------------------------

        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.finetune_lr
        )

        model.train()

        adapt_start = time.perf_counter()

        for _ in range(self.finetune_steps):

            logits = model(support_x)

            loss = self.criterion(logits, support_y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        adaptation_time = time.perf_counter() - adapt_start

        # ------------------------------------------------
        # Evaluate on the query set
        # ------------------------------------------------

        model.eval()

        with torch.no_grad():

            query_logits = model(query_x)

            query_loss = self.criterion(query_logits, query_y)

            preds = query_logits.argmax(dim=1)

            accuracy = (preds == query_y).float().mean()

        # ------------------------------------------------
        # Restore pretrained state (rebuild original head
        # shape first so the state_dict fits)
        # ------------------------------------------------

        model.network[self.head_layer_index] = nn.Linear(
            in_features, original_out
        ).to(self.device)

        model.load_state_dict(original_state)

        return {
            "loss": query_loss.detach(),
            "accuracy": accuracy.detach(),
            "logits": query_logits.detach(),
            "predictions": preds.detach(),
            "adaptation_time": adaptation_time,
        }