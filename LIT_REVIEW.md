# Literature Review — FNN_RFI

Purpose
- Collect and summarize prior work on fractal neural architectures, Fibonacci-inspired network design, and ML approaches to Radio Frequency Interference (RFI) detection/mitigation. Goal: identify gaps and a clear novelty claim for the thesis.

Search strategy (high priority queries)
- "FractalNet" neural network
- "fractal neural network" architecture
- "Fibonacci neural network" OR "Fibonacci architecture" OR "Fibonacci scaling"
- "RFI detection" machine learning
- "radio frequency interference" deep learning
- "AOFlagger" RFI
- "RFI dataset" radio astronomy

Initial topics to cover (each entry: papers, summary, methods, datasets used, gaps)
- Fractal architectures
  - Look for: "FractalNet", recursive/fractal modules, multi-branch fractal blocks, and work using self-similar structures.
  - Summarize: architecture design patterns, training strategies (drop-path, stochastic depth, residuals), reported benefits (optimization, generalization), and limitations.

- Fibonacci / natural scaling in networks
  - Look for: explicit use of Fibonacci ratios for widths, layer counts, receptive fields, or multi-branch counts.
  - Summarize evidence (empirical or theoretical) that Fibonacci-based scaling yields benefits.

- Related architectural motifs
  - DenseNet, ResNet, Inception, recursive nets, fractal expansions — compare and contrast to proposed FNN building blocks.

- RFI detection and mitigation
  - Look for: classical flaggers (AOFlagger), threshold/heuristic methods, and ML approaches (CNNs on spectrograms, RNNs for time-series, Transformers, autoencoders, anomaly detection).
  - Note datasets used, labeling approaches (manual, AOFlagger output, simulation), evaluation metrics, and shortcomings.

- Transfer & domain adaptation
  - Look for work on transferring architecture or pretraining from general tasks to radio-astronomy-specific tasks.

What I'll produce next
- A curated list of ~20–40 papers with short summaries and links.
- A short novelty memo identifying where the proposed FNN+transfer-to-RFI fits relative to existing work.

If you want me to fetch and summarise papers, reply "fetch papers" and I'll start retrieving and summarising them (I can prioritise FractalNet, key RFI ML papers, and AOFlagger resources).
