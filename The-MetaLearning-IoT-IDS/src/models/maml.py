"""
MAML Network

Simple MLP used for First-Order MAML.

Unlike ProtoNet, this network outputs class logits directly.
"""

import torch.nn as nn


class MAML(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_classes=5,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(hidden_dim, num_classes),

        )

    def forward(self, x):

        return self.network(x)