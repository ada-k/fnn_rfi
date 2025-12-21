"""Smoke-test script to verify baseline and FNN models run forward on CPU.

Runs a single forward pass for each model with random data and prints parameter
counts and output shapes. Intended for quick validation on a laptop.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from models.fnn import FNN
from models.cnn import SimpleCNN
from models.rnn import LSTMClassifier
from models.transformer import TinyVisionTransformer


def stats(name, model, x):
    import math
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    y = model(x)
    print(f"{name}: params={params:,} out_shape={y.shape}")


def main():
    device = torch.device("cpu")
    x = torch.randn(2, 3, 32, 32).to(device)

    fnn = FNN(in_ch=3, num_classes=10, stages=3, base_channels=8, use_fib=True).to(device)
    cnn = SimpleCNN(in_ch=3, num_classes=10, base_channels=8).to(device)
    rnn = LSTMClassifier(in_ch=3, hidden_size=128, num_layers=1, num_classes=10).to(device)
    vit = TinyVisionTransformer(in_ch=3, img_size=32, patch_size=4, emb_dim=64, depth=2, num_heads=4, num_classes=10).to(device)

    stats('FNN', fnn, x)
    stats('CNN', cnn, x)
    stats('RNN(LSTM)', rnn, x)
    stats('Transformer', vit, x)


if __name__ == '__main__':
    main()
