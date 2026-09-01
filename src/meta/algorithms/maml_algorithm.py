"""
First-Order MAML (FOMAML) Algorithm

Implements the official MAML optimisation loop using the `higher`
library:

  1. Clone the model into a functional, differentiable copy (fmodel)
     whose initial fast weights ARE connected to the real model
     parameters (copy_initial_weights=False — this is the setting
     most broken MAML implementations get wrong).
  2. Adapt fmodel on the support set for `inner_steps` SGD updates.
  3. Evaluate the adapted fmodel on the query set.
  4. Compute the meta-gradient as the gradient of the query loss
     with respect to the ADAPTED (fast) parameters — this is the
     textbook first-order approximation (Finn et al., 2017, Section
     5.2): it ignores second derivatives through the inner-loop
     trajectory and treats the last inner step's gradient as the
     meta-gradient direction.
  5. Copy that gradient onto the real model's parameters and step
     the meta-optimizer directly.

Because step 5 must happen inside the algorithm (the meta-gradient
does not live on `model.parameters()` via ordinary autograd — it has
to be copied there manually), this algorithm owns its own
zero_grad/backward/step and tells MetaTrainer to skip its generic
optimizer step via `skip_optimizer_step=True` in the returned dict.

Set `first_order=False` to instead run full second-order MAML
(true Finn et al. MAML, no approximation) — useful as a sanity-check
baseline against the first-order version if you want one.
"""

import torch
import torch.nn as nn
import higher


class MAMLAlgorithm:

    def __init__(
        self,
        meta_optimizer,
        inner_lr=0.01,
        inner_steps=5,
        first_order=True,
        device="cpu",
    ):

        self.meta_optimizer = meta_optimizer

        self.inner_lr = inner_lr

        self.inner_steps = inner_steps

        self.first_order = first_order

        self.device = device

        self.criterion = nn.CrossEntropyLoss()

    # ==========================================================
    # Shared inner-loop adaptation
    # ==========================================================

    def _inner_adapt(self, fmodel, diffopt, support_x, support_y):

        for _ in range(self.inner_steps):

            support_logits = fmodel(support_x)

            support_loss = self.criterion(support_logits, support_y)

            diffopt.step(support_loss)

    # ==========================================================
    # Training Step (one episode == one meta-gradient step)
    # ==========================================================

    def training_step(self, model, support_x, support_y, query_x, query_y):

        model.train()

        self.meta_optimizer.zero_grad()

        inner_opt = torch.optim.SGD(model.parameters(), lr=self.inner_lr)

        # For first-order MAML we don't need the full unrolled graph
        # back to theta_0 (that's what makes it "first order" / cheap).
        # For full second-order MAML we do need it, plus access to the
        # time=0 (original) fast weights below.
        track_higher_grads = not self.first_order

        with higher.innerloop_ctx(
            model,
            inner_opt,
            copy_initial_weights=False,   # <-- critical: keeps fast
                                           #     weights connected to
                                           #     the real model
            track_higher_grads=track_higher_grads,
        ) as (fmodel, diffopt):

            self._inner_adapt(fmodel, diffopt, support_x, support_y)

            query_logits = fmodel(query_x)

            query_loss = self.criterion(query_logits, query_y)

            preds = query_logits.argmax(dim=1)

            accuracy = (preds == query_y).float().mean()

            if self.first_order:

                # Gradient wrt the ADAPTED (fast) parameters — the
                # first-order approximation.
                grads = torch.autograd.grad(
                    query_loss,
                    list(fmodel.parameters()),
                    retain_graph=False,
                    allow_unused=True,
                )

            else:

                # Full MAML: gradient wrt the ORIGINAL parameters,
                # backpropagated through the entire inner-loop
                # trajectory.
                grads = torch.autograd.grad(
                    query_loss,
                    list(fmodel.parameters(time=0)),
                    retain_graph=False,
                    allow_unused=True,
                )

            # detach logits/preds before leaving the higher context,
            # since fmodel's graph is torn down on exit
            query_logits = query_logits.detach()

            preds = preds.detach()

        # ----------------------------------------------------
        # Copy the meta-gradient onto the real model parameters
        # ----------------------------------------------------

        for p, g in zip(model.parameters(), grads):

            if g is not None:

                if p.grad is None:

                    p.grad = g.detach().clone()

                else:

                    p.grad += g.detach()

        self.meta_optimizer.step()

        return {
            "loss": query_loss.detach(),
            "accuracy": accuracy.detach(),
            "logits": query_logits,
            "predictions": preds,
            "skip_optimizer_step": True,
        }

    # ==========================================================
    # Validation Step (no meta-update, just adapt + evaluate)
    # ==========================================================

    def validation_step(self, model, support_x, support_y, query_x, query_y):

        model.eval()

        inner_opt = torch.optim.SGD(model.parameters(), lr=self.inner_lr)

        with torch.enable_grad():

            with higher.innerloop_ctx(
                model,
                inner_opt,
                copy_initial_weights=False,
                track_higher_grads=False,   # no backprop needed at all
            ) as (fmodel, diffopt):

                self._inner_adapt(fmodel, diffopt, support_x, support_y)

                query_logits = fmodel(query_x)

                query_loss = self.criterion(query_logits, query_y)

                preds = query_logits.argmax(dim=1)

                accuracy = (preds == query_y).float().mean()

                # detach before leaving the higher context
                query_logits = query_logits.detach()

                preds = preds.detach()

                import time  #time

                adapt_start = time.perf_counter()

                self._inner_adapt(fmodel, diffopt, support_x, support_y)

                adaptation_time = time.perf_counter() - adapt_start

        return {
            "loss": query_loss.detach(),
            "accuracy": accuracy.detach(),
            "logits": query_logits,
            "predictions": preds,
            "adaptation_time": adaptation_time,
        }