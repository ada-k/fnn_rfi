"""RNN/LSTM baseline for sequence-like inputs.

The input spectrogram (B, C, H, W) is reshaped so that frequency rows become
time steps and W*C becomes the per-step feature dimension. LazyLinear is used
for the projection so the module works with any input spatial size.
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM-based classifier that accepts 2D image inputs of any spatial size.

    Reshapes (B, C, H, W) → (B, H, W*C) so rows are treated as time steps.
    LazyLinear infers the per-step feature dimension on the first forward pass,
    making the module independent of the input width.
    """

    def __init__(self, in_ch: int = 3, hidden_size: int = 128, num_layers: int = 1, num_classes: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LazyLinear infers in_features on first call — works for any image width
        self.project = nn.LazyLinear(hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Treat frequency rows as sequence steps; W*C is the per-step feature
        seq = x.permute(0, 2, 3, 1).reshape(b, h, w * c)
        seq = self.project(seq)
        out, _ = self.lstm(seq)
        last = out[:, -1, :]  # last time step
        return self.classifier(last)


if __name__ == "__main__":
    m = LSTMClassifier()
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(y.shape)
