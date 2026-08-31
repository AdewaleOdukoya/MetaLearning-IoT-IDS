"""
Prototypical Network Encoder

This class ONLY defines the embedding network.
Training logic is implemented separately.
"""

import torch.nn as nn


class ProtoNet(nn.Module):

    def __init__(

        self,

        input_dim,

        embedding_dim=128,

    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 128),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, embedding_dim),

        )

    def forward(self, x):

        return self.encoder(x)