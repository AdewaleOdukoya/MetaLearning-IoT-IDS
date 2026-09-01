"""
Reptile Network

Simple MLP used for Reptile (Nichol et al., 2018).

Outputs class logits directly, same as MAML — Reptile does not
require any special architecture; any standard classifier works,
since the inner loop is just ordinary gradient descent.
"""

import torch.nn as nn


class Reptile(nn.Module):

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