# Literature Review — Summaries of downloaded papers

This file contains concise, actionable summaries of the papers already placed in `fnn_rfi/data/papers/`.

1) FractalNet: Ultra-Deep Neural Networks without Residuals
 - Citation: Larsson, Maire, Shakhnarovich, arXiv:1605.07648 (ICLR 2017)
 - Summary: Introduces the FractalNet macro-architecture built from repeated expansion rules that produce truncated fractal layouts. The design creates multiple interacting subpaths of different lengths; training uses a novel regulariser "drop-path" (extension of dropout) to prevent co-adaptation. FractalNet demonstrates that very deep architectures can succeed without residual connections by allowing a training-time transition from effectively shallow to deep.
 - Methods: recursive multi-branch blocks, drop-path regularisation, anytime-depth property (extract shallow/deep subnetworks), experiments on CIFAR/ImageNet.
 - Datasets: CIFAR-10/100 and ImageNet (reported in the paper).
 - Key results: competitive performance with ResNets on common benchmarks; insights about path diversity and training dynamics.
 - Relevance: Core inspiration for FNN design. Key takeaways to use: recursive block construction, drop-path regularisation, and the utility of interacting paths of varying depth.

2) Deep Learning improves identification of Radio Frequency Interference
 - Citation: Vafaei Sadr et al., arXiv:2005.08992 (MNRAS 2020)
 - Summary: Demonstrates that convolutional deep-learning models can improve RFI identification over classical heuristics on real radio-astronomy data. The paper frames RFI-flagging as a supervised classification/segmentation task on time-frequency data (dynamic spectra) and evaluates performance gains from learned features.
 - Methods: CNN-based classifiers/segmenters on spectrogram/dynamic-spectrum inputs; comparison to classical flagging methods (thresholding/AOFlagger-derived baselines); evaluation with precision/recall/F1.
 - Datasets: Observational radio datasets used in the paper (see PDF for exact telescope/dataset details); labeling often relies on AOFlagger outputs or manually curated flags.
 - Key results: Significant improvements in detection metrics in many tested scenarios; analysis of false positives and domain generalisation.
 - Relevance: Provides baselines, dataset handling patterns, and evaluation practices for RFI detection tasks. Use as a baseline comparison and for designing evaluation metrics and label-generation pipelines.

3) Real-Time RFI Mitigation for the Apertif Radio Transient System
 - Citation: Sclocco et al., arXiv:2001.03389 (2020)
 - Summary: Describes an operational, low-latency RFI mitigation pipeline designed for the Apertif radio transient search system. Focuses on architecture for real-time flagging and mitigation within a streaming/production pipeline and evaluates processing constraints and latency.
 - Methods: engineering-focused pipeline design, combination of real-time heuristics and integration with existing flagging tools; discussion of system throughput and integration with transient detection stages.
 - Datasets: Apertif observational datasets used for pipeline testing (see paper for details).
 - Key results: Demonstrates feasibility and performance characteristics of real-time RFI flagging for transient searches; documents constraints to consider when deploying ML-based methods in production pipelines.
 - Relevance: Important operational reference — highlights latency constraints, integration concerns, and what production-grade RFI mitigation must satisfy (useful when transferring FNN to real telescope pipelines).

Next steps
- I'll summarise the remaining downloaded papers (if any) and then continue to fetch more high-priority RFI ML papers and fractal/Fibonacci architecture literature. I'll add these summaries to this file as I go.

4) Detecting HI Galaxies with Deep Neural Networks in the Presence of Radio Frequency Interference
 - Citation: Liang et al., arXiv:2304.13108 (RAA / 2023)
 - Summary: Applies deep neural networks to detect HI galaxies in survey data contaminated by RFI. Focuses on maintaining detection sensitivity while reducing false positives arising from RFI.
 - Methods: CNN-based detection pipelines with preprocessing to handle dynamic spectra and masking strategies; evaluation includes injecting RFI or using flagged observations.
 - Datasets: Survey datasets used in the study (see PDF for telescope and observation details).
 - Key results: Improved detection rates of HI sources when using deep-learning based pre-processing/flagging; discusses trade-offs between sensitivity and false-positive rate.
 - Relevance: Demonstrates the downstream impact of effective RFI mitigation on scientific tasks (source detection), motivating robust flagging methods like the proposed FNN.

5) RFI-DRUnet: Restoring dynamic spectra corrupted by radio frequency interference
 - Citation: Zhang, Cognard, Dobigeon, arXiv:2402.13867 (2024)
 - Summary: Proposes RFI-DRUnet, a restoration network for dynamic spectra corrupted by RFI. Emphasises recovery of underlying astrophysical signals rather than only binary flagging.
 - Methods: U-Net-like architecture with dynamic-spectrum-specific modules and loss functions tailored to signal reconstruction; demonstrates restoration on pulsar and transient observation types.
 - Datasets: Pulsar observations and simulated RFI case studies (see paper for specifics).
 - Key results: Shows qualitative and quantitative improvements in restored spectra and downstream pulsar search metrics.
 - Relevance: Complements detection/flagging approaches by restoring affected data — an alternative objective for ML methods that the FNN could be compared against.

6) Radio Frequency Interference Detection Using Efficient Multi-Scale Convolutional Attention UNet
 - Citation: Gu et al., arXiv:2404.06055 (2024)
 - Summary: Presents an attention-augmented multi-scale UNet tailored for RFI detection in dynamic spectra. Focuses on capturing multi-scale RFI signatures and improving segmentation granularity.
 - Methods: Multi-scale convolutional encoder-decoder with attention modules and tailored loss functions for segmentation; comparisons to baseline UNet/CNN models.
 - Datasets: Observational spectrogram datasets used in the paper (see PDF).
 - Key results: Improved segmentation metrics (precision/recall/F1) and better localization of narrowband/broadband RFI.
 - Relevance: Provides a strong segmentation baseline (UNet + attention) for our comparisons when adapting FNN to RFI detection.

7) RF Challenge: The Data-Driven Radio Frequency Signal Separation Challenge
 - Citation: Lancho et al., arXiv:2409.08839 (2024)
 - Summary: Documents a community challenge on RF signal separation and interference rejection. Includes dataset details and baseline results for signal separation tasks relevant to interference mitigation.
 - Methods: Diverse set of ML approaches evaluated for signal separation tasks; the challenge highlights dataset curation, metrics, and common pitfalls (leakage, overfitting).
 - Datasets: Public challenge datasets and splits; useful as a benchmark for RF separation/mitigation performance.
 - Key results: Benchmarks for participant methods; discussion on dataset bias and evaluation robustness.
 - Relevance: Useful benchmark for broader RF interference tasks; some methods and metrics translate to radio-astronomy RFI mitigation and can inform evaluation protocols.

8) document.pdf (unspecified)
 - Note: a file named `document.pdf` is present in `fnn_rfi/data/papers/` but lacks a descriptive filename. Please confirm which paper this is (I can auto-extract title if you want me to open it). If it is the Segment-Anything applied to RFI paper (arXiv:2410.22497) or another entry, I'll summarise it similarly.


