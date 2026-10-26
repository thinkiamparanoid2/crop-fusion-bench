"""Shared model definitions for E1/E2/E3.

A single E1MLP definition used both by E1 training and by the E2/E3 frozen IoT
encoder, so the encoder architecture cannot drift between experiments.
"""

import torch.nn as nn


class E1MLP(nn.Module):
    """IoT-only MLP classifier.

    Trained standalone for E1, then frozen and reused as the IoT encoder for
    E2/E3. ``embedding(x)`` returns the hidden2-dimensional representation after
    the second ReLU (the 64-dim vector consumed by the fusion head). Dropouts
    default to 0.0 so the canonical E1 config and the frozen-encoder use match.
    """

    def __init__(self, input_dim, num_classes, hidden1=128, hidden2=64, dropout1=0.0, dropout2=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def embedding(self, x):
        for layer in list(self.net.children())[:7]:
            x = layer(x)
        return x
