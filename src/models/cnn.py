"""
1D Convolutional Neural Network (CNN)

Baseline Model 2

Designed for tabular IoT intrusion detection data.

Input:
    (Batch Size, 46)

Internally reshaped to

    (Batch Size, 1, 46)
"""

import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes,
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                num_classes,
            )

        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.features(x)

        x = self.classifier(x)

        return x