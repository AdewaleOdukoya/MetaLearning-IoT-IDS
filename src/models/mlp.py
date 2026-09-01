"""
Multi-Layer Perceptron (MLP) Baseline Model

This model serves as the first baseline for comparison with
CNN, CNN-LSTM, and Meta-Learning models.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Network Intrusion Detection.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim2, num_classes)

        )

    def forward(self, x):
        return self.network(x)