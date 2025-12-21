# FNN Specification — design notes

Goal
- Define a Fractal Neural Network (FNN) family to be implemented, tested on synthetic tasks, and transferred to RFI detection.

Core idea (high level)
- Build the network from recursive/fractal building blocks that embed self-similarity. Use Fibonacci-inspired scaling rules to determine branch counts or channel-width scaling so the architecture reflects natural growth patterns.

Concrete design options (to experiment with)
- Option A — Fractal multi-branch block:
  - A block contains multiple parallel sub-branches arranged in a fractal pattern. Each recursive level composes two or more smaller fractal blocks.
  - Use Fibonacci sequence for branch counts across levels (e.g., levels with 1, 1, 2, 3 branches) or for relative channel widths (scale channels by Fibonacci numbers or golden ratio φ≈1.618).

- Option B — Fibonacci-width stacking:
  - Stack layers where the number of filters/channels follows Fibonacci numbers (e.g., 8, 13, 21, 34) or scaled versions thereof to control parameter count.

- Option C — Recursive residual fractal:
  - Combine fractal recursion with residual connections (or drop-path) to stabilise training, similar to FractalNet variants.

Low-level modules
- Convolutional branch: 1–3 conv layers (kernel sizes 3,5), batchnorm, nonlinear (ReLU/Swish).
- Merge rules: concat or sum followed by 1x1 conv to reduce channels.
- Regularisation: drop-path, stochastic depth, weight decay.

Training & tests (before RFI)
- Synthetic tests: CIFAR10/CIFAR100 or small speech/spectrogram toy tasks to validate training; measure convergence speed, generalization, and robustness.
- Ablations: remove Fibonacci scaling, replace fractal block with simple stacked convs, vary depth/width.

Hyperparameters (initial)
- Optimizer: AdamW or SGD+momentum
- LR schedule: Cosine annealing or step decay
- Batch sizes: 64/128 depending on memory

Deliverables from this spec
- Minimal, working PyTorch implementation of FNN blocks with configurable options.
- Small experiment notebook demonstrating training behaviour and comparing to a simple CNN baseline.
