"""
Reptile Algorithm (Nichol et al., 2018)

Unlike MAML, Reptile does NOT differentiate through the inner-loop
optimisation trajectory, so it needs no `higher` functional wrapper.
The procedure per task/episode is:

  1. Snapshot the current meta-parameters (theta_old).
  2. Adapt the REAL model in-place for `inner_steps` ordinary SGD
     steps on the support set (phi = SGD(theta_old, support_data)).
  3. Evaluate the adapted model on the query set (for logging only
     — this does NOT feed into the meta-update, unlike MAML's query
     loss).
  4. Restore the model to theta_old.
  5. Treat g = theta_old - phi as a pseudo-gradient and hand it to
     the meta-optimizer, which then steps theta toward phi. Using
     Adam (rather than plain SGD) here is what the Reptile paper
     itself recommends — it lets momentum/adaptive scaling smooth
     out the noisy per-task pseudo-gradient direction.

Like MAMLAlgorithm, this class owns its own zero_grad/backward/step
and signals MetaTrainer to skip its generic optimizer step via
`skip_optimizer_step=True`.
"""

import copy
import time

import torch
import torch.nn as nn


class ReptileAlgorithm:

    def __init__(
        self,
        meta_optimizer,
        inner_lr=0.01,
        inner_steps=5,
        device="cpu",
    ):

        self.meta_optimizer = meta_optimizer

        self.inner_lr = inner_lr

        self.inner_steps = inner_steps

        self.device = device

        self.criterion = nn.CrossEntropyLoss()

    # ==========================================================
    # Shared inner-loop adaptation (trains the REAL model in place)
    # ==========================================================

    def _inner_adapt(self, model, support_x, support_y):

        inner_opt = torch.optim.SGD(model.parameters(), lr=self.inner_lr)

        for _ in range(self.inner_steps):

            support_logits = model(support_x)

            support_loss = self.criterion(support_logits, support_y)

            inner_opt.zero_grad()

            support_loss.backward()

            inner_opt.step()

    # ==========================================================
    # Training Step (one episode == one Reptile meta-update)
    # ==========================================================

    def training_step(self, model, support_x, support_y, query_x, query_y):

        model.train()

        # ------------------------------------------------
        # 1. Snapshot theta_old
        # ------------------------------------------------

        original_state = copy.deepcopy(model.state_dict())

        # ------------------------------------------------
        # 2. Adapt in-place on the support set -> phi
        # ------------------------------------------------

        self._inner_adapt(model, support_x, support_y)

        # ------------------------------------------------
        # 3. Evaluate adapted model on the query set
        #    (logging/monitoring only — not used in the
        #    meta-update itself)
        # ------------------------------------------------

        model.eval()

        with torch.no_grad():

            query_logits = model(query_x)

            query_loss = self.criterion(query_logits, query_y)

            preds = query_logits.argmax(dim=1)

            accuracy = (preds == query_y).float().mean()

        adapted_state = copy.deepcopy(model.state_dict())

        # ------------------------------------------------
        # 4. Restore theta_old before applying the meta-step
        # ------------------------------------------------

        model.load_state_dict(original_state)

        # ------------------------------------------------
        # 5. Reptile pseudo-gradient: g = theta_old - phi
        # ------------------------------------------------

        self.meta_optimizer.zero_grad()

        for name, p in model.named_parameters():

            p.grad = (
                original_state[name] - adapted_state[name]
            ).clone()

        self.meta_optimizer.step()

        return {
            "loss": query_loss.detach(),
            "accuracy": accuracy.detach(),
            "logits": query_logits.detach(),
            "predictions": preds.detach(),
            "skip_optimizer_step": True,
        }

    # ==========================================================
    # Validation Step (adapt on a COPY, never touch real weights)
    # ==========================================================

    def validation_step(self, model, support_x, support_y, query_x, query_y):

        # Must snapshot and restore, exactly like training_step,
        # otherwise every validation episode would permanently
        # perturb the meta-parameters.

        original_state = copy.deepcopy(model.state_dict())

        model.train()

        adapt_start = time.perf_counter()

        self._inner_adapt(model, support_x, support_y)

        adaptation_time = time.perf_counter() - adapt_start

        model.eval()

        with torch.no_grad():

            query_logits = model(query_x)

            query_loss = self.criterion(query_logits, query_y)

            preds = query_logits.argmax(dim=1)

            accuracy = (preds == query_y).float().mean()
            

        # Restore — validation must be a read-only operation
        # from the meta-parameters' point of view.

        model.load_state_dict(original_state)

        return {
            "loss": query_loss.detach(),
            "accuracy": accuracy.detach(),
            "logits": query_logits.detach(),
            "predictions": preds.detach(),
            "adaptation_time": adaptation_time,
        }