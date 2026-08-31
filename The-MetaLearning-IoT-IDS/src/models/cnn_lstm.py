"""
CNN-LSTM Network (reduced for CPU training feasibility)
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes,
    ):

        super().__init__()

        self.cnn = nn.Sequential(

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
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

        )

        self.lstm = nn.LSTM(

            input_size=64,

            hidden_size=64,      # was 128

            num_layers=1,        # was 2

            batch_first=True,

        )

        self.classifier = nn.Sequential(

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, num_classes),

        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.cnn(x)

        x = x.permute(0, 2, 1)

        _, (hidden, _) = self.lstm(x)

        x = hidden[-1]

        x = self.classifier(x)

        return x