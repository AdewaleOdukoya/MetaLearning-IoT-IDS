"""
ProtoNet Algorithm

Contains the optimisation logic for
Prototypical Networks.
"""

import torch

import time

from src.losses.prototypical_loss import PrototypicalLoss


class ProtoNetAlgorithm:

    def __init__(self):

        self.loss_fn = PrototypicalLoss()

    # ======================================================
    # Training
    # ======================================================

    def training_step(

        self,

        model,

        support_x,

        support_y,

        query_x,

        query_y,

    ):

        support_embeddings = model(

            support_x

        )

        query_embeddings = model(

            query_x

        )

        return self.loss_fn(

            support_embeddings,

            support_y,

            query_embeddings,

            query_y,

        )

    # ======================================================
    # Validation
    # ======================================================

    # ======================================================
    # Validation
    # ======================================================

    def validation_step(

        self,

        model,

        support_x,

        support_y,

        query_x,

        query_y,

    ):

        with torch.no_grad():

            # ProtoNet's "adaptation" analogue: embedding the
            # support set (from which prototypes are formed).
            # Timed for comparability with the gradient-based
            # adaptation of MAML / Reptile / fine-tuning.

            adapt_start = time.perf_counter()

            support_embeddings = model(support_x)

            adaptation_time = time.perf_counter() - adapt_start

            query_embeddings = model(query_x)

            result = self.loss_fn(

                support_embeddings,

                support_y,

                query_embeddings,

                query_y,

            )

        result["adaptation_time"] = adaptation_time

        return result