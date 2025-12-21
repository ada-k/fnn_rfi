"""RNN/LSTM baseline for sequence-like inputs.

For image-based tests we reshape images into sequences (e.g., rows as time
steps). This is a lightweight LSTM classifier used as a baseline.
"""
from typing import Tuple

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Simple LSTM-based classifier.

    Expects input of shape (B, C, H, W). We reshape to (B, H, W*C) and treat
    H as sequence length with feature dim W*C.
    """

    def __init__(self, in_ch: int = 3, hidden_size: int = 128, num_layers: int = 1, num_classes: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # We'll project per-step features to hidden_size before LSTM
        self.project = nn.Linear(in_ch * 32, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        # reshape to (B, H, W*C)
        seq = x.permute(0, 2, 3, 1).reshape(b, h, w * c)
        # project each step
        seq = self.project(seq)
        out, (hn, cn) = self.lstm(seq)
        # use last hidden state
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits


if __name__ == "__main__":
    m = LSTMClassifier()
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(y.shape)
