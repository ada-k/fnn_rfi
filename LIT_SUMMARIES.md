# Literature Summaries — initial

1) FractalNet (self-similarity / fractal architectures)
- Link: https://arxiv.org/abs/1605.07648 (FractalNet: Ultra-Deep Neural Networks without Residuals)
- Summary: FractalNet introduces a macro-architecture design based on self-similarity. The network is built by repeated application of an expansion rule that produces truncated fractal layouts. FractalNet contains interacting subpaths of different lengths and uses a regulariser called drop-path (extension of dropout) to avoid co-adaptation. The paper shows FractalNet can match ResNet performance on CIFAR and ImageNet, and highlights properties like anytime-depth and ability to extract high-performance subnetworks.

2) RFI / radio-astronomy ML papers (representative list from arXiv search)
- Deep Learning improves identification of Radio Frequency Interference — https://arxiv.org/abs/2005.08992 (MNRAS, 2020)
- Real-Time RFI Mitigation for the Apertif Radio Transient System — https://arxiv.org/abs/2001.03389 (2020)
- Detecting HI Galaxies with Deep Neural Networks in the Presence of Radio Frequency Interference — https://arxiv.org/abs/2304.13108 (2023)
- RFI-DRUnet: Restoring dynamic spectra corrupted by radio frequency interference — https://arxiv.org/abs/2402.13867 (2024)
- Radio Frequency Interference Detection Using Efficient Multi-Scale Convolutional Attention UNet — https://arxiv.org/abs/2404.06055 (2024)
- (Many other arXiv entries returned by a general query; see the arXiv search page for more: https://arxiv.org/search/?query=radio+frequency+interference+deep+learning)

3) Tools and archives
- LOFAR Long Term Archive (LTA) — https://lta.lofar.eu/ — metadata is queryable anonymously; downloading public data requires registration. Useful for raw spectrograms and LOFAR observations.
- AOFlagger — common RFI flagging tool used to create labels from raw data (search for AOFlagger repo and docs; GitHub and project pages contain usage examples). If automatic retrieval fails, use the AOFlagger project page and documentation to obtain binaries/source.
- NRAO / VLA Archive: https://archive.nrao.edu/ — public datasets for radio astronomy.

4) Initial takeaways and reading priorities
- Read FractalNet fully; its architectural motifs (recursive multi-branch, drop-path, anytime property) are core inspiration for FNN variants.
- Prioritise recent RFI ML papers (2020–2024) that use CNN/U-Net-like architectures and papers using AOFlagger outputs for labels.
- Collect code/data repositories linked from those papers (many provide links to sample datasets or processing pipelines).

5) Next literature steps (planned)
- Fetch and summarise 20–30 prioritized papers: FractalNet variants, any explicit "Fibonacci" architecture papers, and the top RFI-ML papers above.
- Extract datasets used and direct download commands where available.
- Produce a novelty memo linking gaps to our proposed FNN+transfer approach.

If you'd like, I can start fetching the PDFs and write per-paper summaries (abstract, methods, datasets, key results, how it relates to our idea). Reply "fetch papers" to begin automated fetching and summarisation.
