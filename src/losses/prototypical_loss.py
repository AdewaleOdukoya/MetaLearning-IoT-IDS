"""
Prototypical Loss

Computes:

1. Class prototypes
2. Euclidean distances
3. Classification logits
4. Cross entropy loss
5. Episode accuracy
"""

import torch
import torch.nn.functional as F


class PrototypicalLoss:

    def __init__(self):

        pass

    # ==========================================================
    # Compute Class Prototypes
    # ==========================================================

    def compute_prototypes(
        self,
        support_embeddings,
        support_labels,
    ):

        classes = torch.unique(support_labels)

        prototypes = []

        for cls in classes:

            prototype = support_embeddings[
                support_labels == cls
            ].mean(dim=0)

            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)

        return prototypes

    # ==========================================================
    # Pairwise Euclidean Distance
    # ==========================================================

    def euclidean_distance(
        self,
        query_embeddings,
        prototypes,
    ):

        distances = torch.cdist(
            query_embeddings,
            prototypes,
            p=2,
        )

        return distances

    # ==========================================================
    # Forward
    # ==========================================================

    def __call__(
        self,
        support_embeddings,
        support_labels,
        query_embeddings,
        query_labels,
    ):

        prototypes = self.compute_prototypes(

            support_embeddings,

            support_labels,

        )

        distances = self.euclidean_distance(

            query_embeddings,

            prototypes,

        )

        logits = -distances

        loss = F.cross_entropy(

            logits,

            query_labels,

        )

        predictions = torch.argmax(

            logits,

            dim=1,

        )

        accuracy = (

            predictions == query_labels

        ).float().mean()

        return {

            "loss": loss,

            "accuracy": accuracy,

            "prototypes": prototypes,

            "predictions": predictions,

            "logits": logits,

        }