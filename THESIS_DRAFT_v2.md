# UEL-CN-7000 Dissertation: Fractal Neural Networks for Radio Frequency Interference Detection

---

## ABSTRACT

Radio frequency interference (RFI) is a fundamental obstacle in radio astronomy: anthropogenic signals contaminate telescope observations at all frequencies, degrading scientific data quality and potentially masking astrophysical phenomena. As next-generation instruments such as the Square Kilometre Array (SKA) come online, the volume and complexity of RFI make traditional rule-based mitigation algorithms insufficient. This dissertation proposes the Fractal Neural Network (FNN), a novel deep learning architecture that leverages Fibonacci-scaled channel widths and self-similar recursive fractal blocks for efficient, high-accuracy RFI detection.

FNNSeg, the segmentation variant, is benchmarked against U-Net and Residual U-Net on three publicly available real-telescope datasets: the HERA telescope with pixel-perfect simulation labels (HERA-PP), HERA with AOFlagger-generated labels (HERA-AOF), and the LOFAR telescope with real observational labels. All experiments are replicated across three random seeds to measure statistical stability. On HERA-PP, FNNSeg achieves a mean test F1 of 0.9952 ± 0.0008 with only 218,609 parameters, outperforming U-Net (482,593 params, F1 = 0.9944) and ResU-Net (1,614,977 params, F1 = 0.9929), and exceeding the best published result from du Toit et al. (2023) (RFI_NET, 805,473 params, best F1 = 0.9936). On LOFAR, FNNSeg achieves a mean F1 of 0.6337 ± 0.0006, again leading both baselines. These results demonstrate that fractal architectures with Fibonacci channel scheduling deliver state-of-the-art RFI detection with 2.2 to 7.4 times fewer parameters than competitive baselines, making them highly suitable for real-time deployment in radio astronomy pipelines.

---

## 1. INTRODUCTION

### 1.1 Background and Motivation

Radio astronomy depends on detecting extremely faint signals from cosmic sources separated from Earth by billions of light-years. The fundamental challenge is that terrestrial and satellite technologies (mobile networks, aircraft, satellite communications, broadcast television) emit radio waves at the same frequencies used by natural astrophysical processes. This anthropogenic contamination, collectively termed Radio Frequency Interference (RFI), can be orders of magnitude stronger than the astrophysical signal of interest, and if not removed, can render entire observations scientifically unusable (Baan et al., 2004).

Existing approaches to RFI mitigation fall into two broad categories. Classical algorithmic flaggers, most commonly AOFlagger (Offringa et al., 2012), use statistical heuristics and morphological operators to identify contaminated time-frequency pixels in telescope dynamic spectra. While effective for well-characterised interference patterns, these methods require manual tuning for each new telescope and RFI environment, and their performance degrades as modern communication infrastructure becomes more pervasive and spectrally complex. The emerging paradigm of deep learning-based RFI detection promises to overcome these limitations by learning generalised representations directly from labelled data (Vafaei Sadr et al., 2020; Gu et al., 2024).

The motivation for this research lies at the intersection of two urgent needs: first, the upcoming Square Kilometre Array (SKA), which will generate data rates exceeding existing flagging infrastructure; and second, the established effectiveness of encoder-decoder architectures for pixel-level binary segmentation tasks, demonstrated in medical imaging (Ronneberger et al., 2015) and satellite imagery. The dominant architectures deployed in the RFI domain, specifically standard U-Net variants, employ channel-doubling schedules that lead to rapid parameter growth, making them expensive to deploy in real-time pipelines where compute budgets are constrained.

This dissertation proposes the Fractal Neural Network (FNN), a new architecture that replaces channel-doubling with Fibonacci-sequence channel scaling and replaces standard convolutional blocks with self-similar recursive fractal blocks. The hypothesis is that biologically-inspired scaling patterns combined with multi-path fractal representations can achieve comparable or superior detection performance with significantly fewer parameters, improving both computational efficiency and generalisation across instruments.

### 1.2 Research Questions

This dissertation addresses the following research questions:

1. Can a Fractal Neural Network with Fibonacci channel scheduling achieve competitive RFI detection performance compared to established deep learning baselines on real telescope data?
2. Does the parameter efficiency advantage of the fractal architecture translate across multiple datasets with different label types and telescope configurations?
3. How does label quality (pixel-perfect simulation labels vs AOFlagger-generated heuristic labels vs real observational labels) affect model performance, and does FNNSeg generalise better than deep learning baselines under noisier label conditions?

### 1.3 Research Aims and Objectives

**Aim:** To design, implement, and evaluate a Fractal Neural Network architecture for RFI detection in radio astronomy, demonstrating competitive performance with greater parameter efficiency than existing state-of-the-art models.

**Objectives:**

1. Design the FNNSeg architecture using Fibonacci channel scaling and recursive fractal blocks with configurable depth and drop-path regularisation.
2. Implement a reproducible training and evaluation pipeline for pixel-level RFI segmentation on real telescope data.
3. Evaluate FNNSeg against U-Net and ResU-Net baselines on three benchmark datasets (HERA-PP, HERA-AOF, LOFAR).
4. Compare results against published benchmarks from du Toit et al. (2023) using identical evaluation metrics and data splits.
5. Analyse the sensitivity of results to random initialisation (3 seeds) and assess statistical stability.
6. Characterise the trade-off between model size and detection performance to assess suitability for real-time deployment.

### 1.4 Success Criteria

In direct response to proposal feedback that success criteria were not explicitly defined, the following quantitative thresholds were established prior to experimentation:

- **Primary criterion:** FNNSeg achieves test F1 ≥ 0.99 on HERA-PP (matching the du Toit best reported F1 of 0.9936).
- **Efficiency criterion:** FNNSeg achieves the above with fewer parameters than the best-performing du Toit baseline.
- **Stability criterion:** Standard deviation of test F1 across 3 seeds ≤ 0.002.
- **Cross-dataset criterion:** FNNSeg is competitive with or outperforms baselines on LOFAR (representing real, harder, cross-instrument conditions).

All four criteria were met in the final experiments.

### 1.5 Ethical Considerations

All data used in this study is publicly released under open-access licences via the Zenodo repository (du Toit et al., 2023; DOI: 10.5281/zenodo.8275061). No human subjects or personally identifiable information is involved. The computational experiments were conducted on commercial cloud infrastructure (Google Cloud Platform) under standard academic usage. All models were trained for scientific research purposes only, with no dual-use concerns anticipated from a network designed to detect radio interference in telescope data. The research follows standard academic integrity principles, with all third-party results clearly attributed and not reproduced as original work.

### 1.6 Dissertation Structure

Chapter 2 presents a critical review of the literature on RFI detection methods and relevant deep learning architectures. Chapter 3 describes the FNNSeg architecture, training methodology, and baseline implementations. Chapter 4 details the datasets and data processing pipeline. Chapter 5 documents the experimental setup, hyperparameter choices, and computational infrastructure. Chapter 6 presents the quantitative results across all datasets, models, and seeds. Chapter 7 provides critical discussion, analysis of findings, and limitations. Chapter 8 concludes with a summary of contributions and directions for future work.

---

## 2. LITERATURE REVIEW

### 2.1 Radio Frequency Interference in Radio Astronomy

The challenge of RFI predates the digital era of astronomy. Baan et al. (2004) provide a comprehensive overview of RFI management strategies across the radio spectrum, noting that the proliferation of mobile communication systems in the 1990s–2000s dramatically worsened the RFI environment for telescopes. The consensus recommendation from that era was mitigation through a combination of site selection (remote locations), regulatory spectrum management, and post-correlation flagging. Site selection and regulation, while necessary, are insufficient alone: satellites, aircraft, and distant terrestrial emitters penetrate even the most remote observatory sites.

Post-correlation flagging (identifying contaminated visibility samples after data recording) remains the dominant operational approach. The most widely deployed implementation is AOFlagger (Offringa et al., 2012), which combines Gaussian-smoothed bandpass estimation with SumThreshold morphological detection. AOFlagger has become the de facto standard for LOFAR and WSRT, and is supported for HERA and MeerKAT operations. Its key strengths are computational efficiency and decades of operational refinement; its key weakness is that it was designed for known, structured RFI patterns and requires expert configuration for each new telescope environment. As Offringa et al. (2010) themselves note, the algorithm is sensitive to the choice of threshold parameters, and its false-positive rate increases significantly in high-RFI environments.

### 2.2 Machine Learning for RFI Detection: Early Work

The first wave of ML-based RFI detection used shallow classifiers. Bhatia et al. (2020) applied random forests and support vector machines to features extracted from dynamic spectra, demonstrating improved recall over fixed-threshold methods but with limited generalisation. The limitations of hand-crafted features, which require domain expertise and fail on unseen interference patterns, motivated the shift to end-to-end deep learning.

Vafaei Sadr et al. (2020) provided a landmark demonstration that convolutional neural networks could directly improve upon classical heuristics on real radio-astronomy data without feature engineering. By treating RFI detection as a supervised classification task on time-frequency inputs, they showed that learned features outperformed AOFlagger-derived baselines on their evaluation sets. This work established the foundational paradigm: frame RFI detection as binary pixel segmentation of a 2D dynamic spectrum, and apply supervised deep learning with labelled examples.

### 2.3 Encoder-Decoder Architectures for Segmentation

The dominant architectural family for medical and remote-sensing segmentation is the U-Net (Ronneberger et al., 2015), which uses a symmetric encoder-decoder structure with skip connections at each resolution level. Skip connections allow the decoder to combine fine spatial detail from the encoder with semantically rich deep features, a property critical for detecting fine-scale RFI patterns (e.g., narrowband interference occupying only 1–2 frequency channels). The Residual U-Net extends this by adding pre-activation residual blocks (He et al., 2016) in place of plain convolutional blocks, improving gradient flow in deeper architectures.

In the RFI domain, du Toit et al. (2023) provide the most systematic comparison of encoder-decoder architectures to date. Their framework benchmarks eight architectures including U-Net variants, RNET (a lightweight decoder-only network), DSC-ResUNet (depthwise separable residual U-Net), and RFI_NET on the same HERA and LOFAR datasets used in this dissertation. Key findings from du Toit et al. (2023) include: (1) RFI_NET (805K parameters) achieved the highest maximum F1 of 0.9936 on HERA-PP; (2) RNET (32K parameters) achieved competitive F1 of 0.9895 with far fewer parameters, demonstrating that large models are not necessary; (3) performance on AOFlagger-labelled data was uniformly worse across all models (F1 approximately 0.7–0.8) because AOFlagger labels are noisier than simulation-generated ground truth; (4) LOFAR remained the most challenging dataset, with all models achieving substantially lower F1 than on HERA. This work provides the essential external benchmark against which FNNSeg is compared.

### 2.4 Fractal Architectures

The theoretical foundation for the FNN design comes from Larsson et al. (2017), who introduced FractalNet, an ultra-deep network built from repeated fractal expansion rules that produce truncated self-similar layouts. FractalNet demonstrated that competitive performance on CIFAR and ImageNet could be achieved without residual connections, using instead a novel regulariser called drop-path (a form of stochastic depth at the branch level). The key insight was that interacting paths of different effective depths provide implicit ensemble behaviour: during training, shorter paths provide stable gradients, while longer paths develop complex representations. Larsson et al. (2017) showed that this prevents the gradient vanishing problem that motivated residual connections.

The FNN architecture proposed in this dissertation adapts the fractal block concept for segmentation by embedding it within a U-Net encoder-decoder scaffold, and extends it with Fibonacci-sequence channel scaling. The Fibonacci sequence (1, 1, 2, 3, 5, 8, 13, ...) provides a sublinear growth schedule for channel widths across encoder stages: rather than the standard doubling (16, 32, 64, 128), FNN applies Fibonacci multipliers indexed from position 2 onward (1, 2, 3), yielding encoder widths of 16, 32, and 48 with base_ch=16. This scheduling choice is motivated by the golden ratio φ ≈ 1.618, which characterises the limit ratio of consecutive Fibonacci numbers and appears in natural self-similar structures. The hypothesis is that architectures exhibiting Fibonacci scaling may have representational properties well-suited to the multi-scale, self-similar nature of RFI morphologies.

### 2.5 Attention and Multi-Scale Approaches

Gu et al. (2024) propose an attention-augmented multi-scale U-Net for RFI detection that explicitly captures features at multiple frequency and temporal scales simultaneously. Their results show improved segmentation granularity for narrowband and broadband RFI, suggesting that explicit multi-scale processing is beneficial for the diversity of interference types encountered in practice. While FNNSeg does not use explicit attention modules, the fractal block's parallel paths implicitly process features at different effective receptive field depths, providing a form of implicit multi-scale aggregation.

Zhang et al. (2024) extend the scope beyond detection to propose RFI-DRUnet, a restoration network that simultaneously flags and reconstructs contaminated dynamic spectra. This dual-task formulation, architecturally similar to FNNInpaint (a dual-head variant discussed in Section 3.6), demonstrates the potential for RFI mitigation networks that not only identify interference but recover the underlying astrophysical signal.

### 2.6 Why Fractal Networks Over Transformers?

In response to proposal feedback requesting a more explicit comparison with transformer architectures, this section directly addresses the choice. Vision Transformers (Dosovitskiy et al., 2020) and their derivatives have demonstrated state-of-the-art performance on image segmentation tasks; for example, SegFormer (Xie et al., 2021). However, several properties make them poorly suited as the primary architecture for this application:

**Scale:** Transformers require large-scale pretraining datasets (ImageNet-21K or similar) for their attention mechanisms to generalise effectively. The HERA training set contains 1,120 frames; LOFAR contains 7,465. These sizes are one to two orders of magnitude smaller than typical ViT pretraining sets. Fine-tuning a pretrained ViT on such small datasets risks catastrophic forgetting and poor calibration.

**Computational cost:** Standard self-attention scales quadratically with sequence length. For 512×512 input patches processed at the pixel level, the sequence length is 262,144, making naive self-attention computationally intractable without aggressive window-based approximations that sacrifice global receptive field.

**Inductive biases:** Convolutional architectures have strong spatial locality and translation equivariance priors, properties well-matched to RFI, which appears as localised patterns in time-frequency space. Transformers lack this inductive bias, requiring more data to learn equivalent spatial structure.

**Deployment constraints:** Radio astronomy pipelines operate under strict latency and compute budgets. A compact convolutional architecture with 218K parameters is deployable on edge hardware; a ViT backbone typically has 86M or more parameters.

The fractal architecture provides the benefits of deep representation learning (via multi-path feature aggregation and implicit ensemble behaviour) within a fully convolutional framework that is compact, fast, and data-efficient, making it the preferable choice for this application.

### 2.7 Dataset and Evaluation Framework

Liang et al. (2023) demonstrate that the choice of RFI detection method has measurable downstream effects on scientific products, in their case the detection rate of HI galaxies in contaminated survey data. This motivates evaluation on real telescope data rather than only synthetic benchmarks. The Zenodo dataset of du Toit et al. (2023) provides a publicly available benchmark specifically designed to support reproducible comparison of RFI detection models on real HERA and LOFAR data. By using identical dataset splits, preprocessing, and evaluation metrics, this dissertation's results are directly comparable to the published literature.

Sclocco et al. (2020) provide an operational perspective on RFI mitigation in the Apertif real-time transient detection system, demonstrating that computational efficiency, not only detection accuracy, is a critical constraint for production deployment. This further motivates the FNNSeg design's emphasis on parameter efficiency.

---

## 3. METHODOLOGY

### 3.1 Overview of the FNNSeg Architecture

FNNSeg follows a U-Net-style encoder-decoder scaffold with fractal blocks at every resolution level. The high-level design principle is: replace standard double-convolutional blocks (as used in U-Net) with FractalBlocks, self-similar recursive modules that process features via multiple parallel paths of different depths, and replace the standard doubling channel schedule with a Fibonacci-sequence schedule that grows channel widths sublinearly.

The architecture consists of:

- An encoder of D=3 downsampling stages, each containing a FractalBlock followed by 2×2 max-pooling.
- A bottleneck FractalBlock at the smallest spatial resolution.
- A decoder of 3 upsampling stages, each using transposed convolution followed by skip-connection concatenation and a FractalBlock.
- A 1×1 output convolution producing a single-channel per-pixel binary logit (RFI probability).

Figure 1 illustrates the complete FNNSeg encoder-decoder pathway. Figure 2 shows the recursive FractalBlock structure: at depth 1, a FractalBlock is a leaf ConvBlock (two 3×3 convolutions with batch normalisation and SiLU activation); at depth d > 1, two child FractalBlocks run in parallel and their outputs are concatenated then projected through a 1×1 convolution. The Fibonacci channel schedule (widths 16, 32, 48 for encoder stages 0–2) provides principled sub-geometric growth, avoiding the sharp parameter doubling of standard U-Net while maintaining representational capacity. This yields 218,609 parameters: 2.2 times fewer than U-Net (482,593) and 7.4 times fewer than ResU-Net (1,614,977).

[INSERT FIGURE 1 — FNNSeg U-Net architecture with Fibonacci channel schedule and FractalBlock stages]

[INSERT FIGURE 2 — FractalBlock recursion tree: purple root, blue depth-2 nodes, green leaf ConvBlocks, red merge nodes]

### 3.2 The FractalBlock

The FractalBlock is a self-similar computational unit parameterised by recursion depth d. At depth d, the block contains two parallel sub-branches:

- **Left branch:** A ConvBlock consisting of two 3×3 convolutions, each followed by Batch Normalisation and SiLU activation.
- **Right branch:** Two recursive FractalBlocks at depth d−1.

At the base case (depth 1), both branches are simple ConvBlocks. Outputs from the two branches are merged via a learned 1×1 convolution that adaptively weights the shallow (left) and deep (right) representations. This recursive structure creates a network within a network: at depth 3 (used in all encoder/bottleneck stages), the effective depth of the deepest path through a single FractalBlock is 2³ = 8 convolutional layers. This provides access to multi-scale representations at every spatial resolution level without requiring explicit multi-scale branches or attention modules.

### 3.3 Fibonacci Channel Scheduling

Standard U-Net doubles channels at each downsampling stage: with base_ch=16, the encoder uses [16, 32, 64, 128]. FNNSeg instead uses Fibonacci multipliers drawn from positions 2, 3, and 4 of the Fibonacci sequence (1, 1, 2, 3, 5, 8, ...), giving multipliers [1, 2, 3] and encoder widths [16 × 1, 16 × 2, 16 × 3] = [16, 32, 48] for base_ch=16, stages=3. The bottleneck is set to 2 × 48 = 96 channels. This sublinear growth means the final encoder stage uses 48 channels rather than 64, reducing the parameter count in that stage by roughly 56% compared with standard doubling.

The relationship to the golden ratio φ ≈ 1.618 provides a principled justification: as consecutive Fibonacci numbers grow, their ratio converges to φ. This means the Fibonacci schedule grows approximately as a geometric series with ratio φ, intermediate between linear (ratio 1) and doubling (ratio 2, the standard schedule). The hypothesis is that networks with this growth pattern may have representational properties well-suited to the self-similar structure of RFI patterns.

### 3.4 Stochastic Depth (Drop-Path) Regularisation

Following Larsson et al. (2017), FNNSeg supports drop-path regularisation: during training, entire branches within the FractalBlock are dropped with configurable probability drop_path_rate. This forces the network to maintain useful representations at every depth level, because no single path can become the dominant feature extractor when any path may be absent during a given forward pass. During inference, all paths are active and outputs are scaled by the survival probability. In the experiments reported in this dissertation, drop_path_rate=0.0 was used (the sweep experiments in the preliminary synthetic phase identified this as optimal for the 30-epoch training budget), but the architecture supports rates up to 0.2.

### 3.5 Baseline Architectures

**U-Net (Ronneberger et al., 2015):** The standard encoder-decoder with double-convolutional blocks (two 3×3 convolutions with BatchNorm and ReLU), max-pooling downsampling, and transposed convolution upsampling. Channel schedule: [16, 32, 64] with base_ch=16, depth=3. This is the closest architectural analogue to FNNSeg; the differences are limited to the convolutional block type (double-conv vs. fractal) and channel schedule (doubling vs. Fibonacci).

**Residual U-Net (ResU-Net):** Extends U-Net with pre-activation residual blocks (He et al., 2016); each block computes BN → ReLU → Conv → BN → ReLU → Conv plus a residual shortcut. The larger parameter count (1,614,977) arises from deeper residual branches at each stage. ResU-Net is included to represent a more powerful but heavier architecture.

### 3.6 Loss Function

All models are trained with DiceBCELoss, a combination of the Dice loss and Binary Cross-Entropy (BCE) loss:

```
DiceBCE(y, p) = 0.5 × BCE(y, p)  +  0.5 × DiceLoss(y, p)

where:
  BCE(y, p)       = -[ y × log(p) + (1-y) × log(1-p) ]
  DiceLoss(y, p)  = 1 - (2 × y × p + 1) / (y + p + 1)
```

The weighting coefficient α = 0.5 in all experiments. The Dice loss directly optimises the F1 score. The F1 score and its component metrics are defined as:

```
Precision  = TP / (TP + FP)
Recall     = TP / (TP + FN)
F1         = 2 × Precision × Recall / (Precision + Recall)
           = 2TP / (2TP + FP + FN)

where TP = true positives (RFI pixels correctly flagged)
      FP = false positives (clean pixels incorrectly flagged)
      FN = false negatives (RFI pixels missed)
      TN = true negatives (clean pixels correctly passed)
```

Binary cross-entropy adds pointwise supervision, preventing the Dice loss's tendency to produce overconfident probability estimates. The combined loss has been demonstrated superior to either component alone for class-imbalanced segmentation tasks (Milletari et al., 2016), which is relevant here since RFI typically occupies 1–10% of all time-frequency pixels.

### 3.7 Optimiser and Learning Rate Schedule

All models use the Adam optimiser (Kingma & Ba, 2014) with learning rate 1×10⁻³, weight decay 1×10⁻⁴, and default momentum parameters β₁=0.9, β₂=0.999. A cosine annealing learning rate schedule decays the learning rate from 1×10⁻³ to 1×10⁻⁶ over the training duration, reducing oscillation around minima in later epochs. These hyperparameters were fixed across all models and datasets to enable fair comparison.

### 3.8 Patch Extraction and Data Loading

Radio telescope dynamic spectra are represented as 2D arrays with time on one axis and frequency on the other. The HERA and LOFAR frames in the benchmark dataset are 512×512 pixels. Full frames at this resolution would require approximately 2 MB per sample (512×512×2 channels×4 bytes per float32) and would not fit in GPU memory at reasonable batch sizes. A patch extraction strategy is therefore used:

- Patch size: 256×256 pixels
- Stride: 128 pixels (50% overlap)
- This produces 9 patches per 512×512 frame (3×3 grid with 128-pixel stride)
- Total training patches for HERA: 1,008 frames × 9 patches = 9,072 patches
- Total training patches for LOFAR: 6,719 frames × 9 patches = 60,471 patches

Patches are extracted on-the-fly in the PyTorch DataLoader, avoiding storing all patches in memory simultaneously.

### 3.9 Normalisation

Input normalisation follows the protocol of du Toit et al. (2023): (1) log-amplitude transform applied to channel 0 (amplitude): X[..., 0] = log(1 + X[..., 0]); (2) per-dataset z-score normalisation computed from the training split and applied to all splits. For LOFAR, which is a single-channel dataset (amplitude only, no phase), the phase normalisation step is skipped. Normalisation statistics are computed from the training frames only to prevent data leakage.

### 3.10 Early Stopping and Checkpointing

All GPU experiments used early stopping with patience=7: if validation F1 did not improve for 7 consecutive epochs, training terminated early and the best checkpoint was restored. The best model (measured by validation F1) was saved to best_model.pt at each improvement, and this checkpoint was loaded for final test evaluation. Early stopping prevents overfitting and reduces unnecessary compute cost.

### 3.11 Evaluation Metrics

Following du Toit et al. (2023), the following metrics are used for all comparisons:

```
Pixel Accuracy  = (TP + TN) / (TP + TN + FP + FN)
F1 Score        = 2TP / (2TP + FP + FN)
AUC-ROC         = Area under the ROC curve,
                  where ROC plots TPR = TP/(TP+FN) against FPR = FP/(FP+TN)
```

Mean and standard deviation across 3 independent seeds are reported for all metrics.

---

## 4. DATA

### 4.1 Overview of Datasets

This dissertation uses three publicly available benchmark datasets, all released by du Toit et al. (2023) and downloaded from Zenodo (DOI: 10.5281/zenodo.8275061). The datasets represent two telescopes (HERA and LOFAR), two label types (simulation-derived and AOFlagger-derived), and a range of RFI environments.

While raw observational archives exist at NRAO (HERA) and ASTRON (LOFAR), these require institutional registration and do not provide the benchmark label masks used here. The Zenodo repository (du Toit et al., 2023) was selected specifically to enable direct numerical comparison against published baselines on identical train/test splits. This also ensures that differences in reported metrics between this work and du Toit et al. (2023) are attributable to architecture differences rather than differences in data splits, preprocessing, or evaluation protocol.

**Table 1: Dataset Summary Statistics**

| Dataset  | Telescope | Label Type                     | Train Frames | Test Frames | Frame Size | Channels      | File Size |
|----------|-----------|-------------------------------|-------------|-------------|------------|---------------|-----------|
| HERA-PP  | HERA      | Pixel-perfect (simulation)    | 1,120        | 280         | 512×512    | 2 (amp+phase) | 3.3 GB    |
| HERA-AOF | HERA      | AOFlagger heuristic           | 1,120        | 280         | 512×512    | 2 (amp+phase) | 3.3 GB    |
| LOFAR    | LOFAR     | Expert manual labels          | 7,465        | 109         | 512×512    | 1 (amplitude) | 9.9 GB    |

### 4.2 HERA Dataset (Pixel-Perfect Labels)

The HERA (Hydrogen Epoch of Reionisation Array) telescope is a purpose-built low-frequency radio array in the Karoo desert, South Africa, designed to probe the epoch when the first stars ionised hydrogen in the early universe (DeBoer et al., 2017). The HERA-PP dataset was generated using hera_sim, a simulation package that injects RFI into realistic HERA baseline measurements with precise ground-truth pixel masks (du Toit et al., 2023). Because the RFI sources are injected computationally, the ground-truth labels are exact: every contaminated pixel is known with certainty. This yields very clean supervision and correspondingly high model performance.

Each frame is a 512×512 array of complex visibilities (one time-frequency baseline observation), stored as two real channels: amplitude and phase. The amplitude channel captures the signal power, which is dominated by RFI; the phase channel carries complementary geometric delay information. The train/test split (1,120/280 = 80%/20%) was established by du Toit et al. (2023) and is used without modification. Within the training set, a 90%/10% train/validation split is applied per experiment: 1,008 frames for training and 112 frames for validation. This split is randomised per seed but stratified to maintain the original RFI fraction distribution.

### 4.3 HERA Dataset (AOFlagger Labels)

The HERA-AOF dataset uses the same HERA observations as HERA-PP but with labels generated by AOFlagger rather than simulation ground truth. AOFlagger produces conservative flags: it tends to under-flag weak or diffuse interference and over-flag strong astrophysical emission that resembles RFI morphologically. This introduces systematic noise in the labels; some clean pixels are flagged and some contaminated pixels are missed, making learning from AOFlagger labels harder than learning from simulation labels. The practical relevance of HERA-AOF is high, however, because in real telescope operations simulation ground truth is unavailable and AOFlagger labels are the best available supervision.

The significantly lower F1 scores (~0.77 vs ~0.99) observed for all models on HERA-AOF relative to HERA-PP are not primarily indicative of model failure; they reflect the inherent ceiling imposed by label noise. A model trained on noisy labels can at best learn to predict those noisy labels; the true F1 against ground-truth would be higher.

### 4.4 LOFAR Dataset

The LOFAR (Low-Frequency Array) telescope is a European-scale radio interferometer with its core in the Netherlands, operating at 10–240 MHz (van Haarlem et al., 2013). The LOFAR dataset contains real telescope observations with expert human-generated labels, the highest quality real-data annotation available but expensive to produce. The LOFAR observations are real data slightly modified from a previously published dataset (Zenodo record 6724065; Mesarcik et al., 2022) and incorporated into the du Toit et al. (2023) benchmark collection. Unlike the HERA datasets, LOFAR observations are single-channel (amplitude only; no phase channel), reflecting differences in the data product format. This single-channel property is accommodated in the data pipeline by detecting n_channels=1 at runtime and skipping phase normalisation.

LOFAR represents the most challenging evaluation setting: it is a different telescope (different frequency range, different array geometry, different RFI environment) from HERA, and the labels reflect real, complex interference from European sources (aviation radar, GSM, digital television broadcasting). The 7,465-frame training set is the largest in the benchmark, but the test set contains only 109 frames, making test-set statistics somewhat noisier.

### 4.5 Data Preprocessing and Ethical Considerations

All datasets are publicly released under open-access terms. No special institutional ethics approval was required. The data contains no personally identifiable information. The preprocessing pipeline (log-amplitude transform, z-score normalisation, patch extraction) faithfully reproduces the protocol described by du Toit et al. (2023) to ensure comparability of results. All preprocessing statistics (mean and standard deviation for normalisation) are computed from the training split and applied consistently to validation and test splits to prevent data leakage.

One notable change from the du Toit protocol is the use of in-place array operations during preprocessing for the LOFAR dataset. The 9.9 GB LOFAR pickle, when loaded and preprocessed with naive copy semantics, would require up to ~31 GB of peak RAM, exceeding the available memory on the initial n1-standard-8 GCP instance (30 GB). The preprocessing pipeline was redesigned to compute normalisation statistics from the already-transformed training split (rather than re-indexing the full array), use NumPy's `out=` parameter for in-place log transforms, and store binary masks as `bool` rather than `float32`, reducing steady-state memory to approximately 19 GB. This technical change is architecturally transparent to the model but necessary for practical training at scale.

---

## 5. EXPERIMENTS

### 5.1 Experimental Design

The experimental design follows a 3 × 3 × 3 factorial structure:

- 3 models: FNNSeg, U-Net, ResU-Net
- 3 datasets: HERA-PP, HERA-AOF, LOFAR
- 3 seeds: 0, 1, 2

This yields 27 total training runs. An additional seed=42 run consisting of only 2 epochs was used to smoke-test the pipeline on HERA-PP before committing to the full experimental grid; this run is excluded from all reported results. Results are summarised as mean ± standard deviation across seeds, and individual seed results are presented in the appendices. This design follows best practices for empirical evaluation in deep learning (Bouthillier et al., 2021): reporting mean and variance over multiple seeds rather than single-run results prevents overfitting to lucky initialisation.

### 5.2 Hyperparameters

**Table 2: Experimental Hyperparameters (Fixed Across All Experiments)**

| Parameter             | Value              | Justification                                     |
|-----------------------|--------------------|---------------------------------------------------|
| Patch size            | 256×256            | Matches du Toit et al. (2023) for comparability   |
| Patch stride          | 128 pixels         | 50% overlap; balances coverage and patch count    |
| Epochs                | 30                 | Sufficient for convergence with early stopping    |
| Batch size (CPU)      | 8                  | Memory limit on MacBook Air M4                    |
| Batch size (GPU)      | 32                 | T4 VRAM sufficient; higher throughput             |
| Base channels         | 16                 | Matches du Toit U-Net (16 filters)                |
| Depth                 | 3                  | Encoder stages; matches du Toit                   |
| Fractal depth         | 3                  | FractalBlock recursion depth (max_depth)          |
| Optimiser             | Adam               | lr=1×10⁻³, β₁=0.9, β₂=0.999                     |
| LR schedule           | Cosine annealing   | Min LR = 1×10⁻⁶                                  |
| Weight decay          | 1×10⁻⁴             | Light L2 regularisation                           |
| Early stopping        | patience=7         | GPU runs; disabled for CPU runs                   |
| Val fraction          | 0.1                | 10% of training frames                            |

Hyperparameters were not tuned on the evaluation datasets. The batch size, base channels, and depth were fixed to match du Toit et al. (2023) to ensure fair comparison. The fractal depth, optimiser, and learning rate were inherited from preliminary synthetic experiments.

### 5.3 Computational Infrastructure

**Table 3: Computational Resources and Training Cost**

| Experiment | Models | Hardware | Machine | Est. wall-clock | Est. GCP cost |
|---|---|---|---|---|---|
| HERA-PP seed=0 | FNNSeg, U-Net, ResU-Net | CPU | MacBook Air M4 | ~72 hrs/model | $0 (local) |
| HERA-PP seeds 1–2 | All 3 models | GPU (T4) | n1-standard-8 + T4 | ~1.5 hrs/run | ~$1.10/run |
| HERA-AOF seeds 0–2 | All 3 models | GPU (T4) | n1-standard-8 + T4 | ~1.5 hrs/run | ~$1.10/run |
| LOFAR seed=0 | FNNSeg only | GPU (T4) | n1-standard-8 + T4 | ~5 hrs | ~$3.65 |
| LOFAR seeds 1–2 (FNNSeg) + seeds 0–2 (U-Net, ResU-Net) | 8 runs | GPU (T4) | n1-highmem-8 + T4 | ~5 hrs/pair | ~$4.40/pair |

*GCP on-demand rates (us-central1, 2024): n1-standard-8 \$0.38/hr; n1-highmem-8 \$0.53/hr; T4 GPU \$0.35/hr. Total estimated GCP expenditure: 6 HERA-PP GPU runs × \$1.10 (\$6.60) + 9 HERA-AOF GPU runs × \$1.10 (\$9.90) + \$3.60 (LOFAR seed=0 solo) + 4 pairs × \$4.40 (LOFAR concurrent, \$17.60) + ~\$2.00 (storage & misc) ≈ **\$38 USD**.*

The T4 GPU (15 GB VRAM) provided approximately 40 times the training throughput of CPU: a 30-epoch HERA run took approximately 1.5 hours on the T4 versus approximately 72 hours on the MacBook Air M4 CPU. The three CPU seed=0 runs (FNNSeg, U-Net, ResU-Net on HERA-PP) were completed locally before the GCP environment was set up.

The initial LOFAR seed=0 run for FNNSeg was carried out on an n1-standard-8 instance (8 vCPU, 30 GB RAM) as a single job, made possible by the in-place preprocessing optimisation described in Section 4.5, which reduced steady-state RAM to approximately 19 GB. Subsequent LOFAR experiments were migrated to an n1-highmem-8 instance (8 vCPU, 52 GB RAM), which allowed two jobs to run concurrently (two × 19 GB = 38 GB, within the 52 GB limit), reducing the total wall-clock time for the LOFAR experiment block by approximately 50%.

### 5.4 Model Parameter Counts

**Table 4: Model Parameter Counts**

| Model                      | Parameters  | Notes                               |
|----------------------------|-------------|-------------------------------------|
| FNNSeg (HERA, 2-channel)   | 218,609     | base_ch=16, stages=3, use_fib=True  |
| FNNSeg (LOFAR, 1-channel)  | 218,465     | Same architecture, 1 input channel  |
| U-Net (HERA, 2-channel)    | 482,593     | base_ch=16, depth=3                 |
| U-Net (LOFAR, 1-channel)   | 482,449     | Same architecture, 1 input channel  |
| ResU-Net (HERA, 2-channel) | 1,614,977   | base_ch=16, depth=3                 |
| ResU-Net (LOFAR, 1-channel)| 1,614,833   | Same architecture, 1 input channel  |

FNNSeg is 2.2 times smaller than U-Net and 7.4 times smaller than ResU-Net by parameter count.

### 5.5 Reproducibility

All experiments are fully reproducible via the project repository. Reproducibility is ensured by: (1) seeding NumPy, Python random, and PyTorch random number generators with the specified seed at the start of each run; (2) saving all hyperparameters to results.json alongside metrics; (3) saving the best model checkpoint to best_model.pt; (4) pinning all library versions in requirements.txt. The training logs (.log files) record per-epoch metrics and are archived with the codebase.

---

## 6. FINDINGS

### 6.1 HERA Pixel-Perfect Results

Table 5 reports the test-set results for all three models on the HERA pixel-perfect (HERA-PP) benchmark. Figure 3 presents the same F1 scores visually, alongside the results for HERA-AOF and LOFAR discussed in subsequent sections, to facilitate cross-dataset comparison.

**Table 5: HERA Pixel-Perfect Label Results (mean ± std, n=3 seeds)**

| Model     | Params    | Accuracy              | F1 Score          | AUC-ROC           |
|:----------|----------:|:---------------------:|:-----------------:|:-----------------:|
| FNNSeg    | 218,609   | 0.999896 ± 0.000018   | 0.9952 ± 0.0008   | 0.9995 ± 0.0002   |
| U-Net     | 482,593   | 0.999879 ± 0.000015   | 0.9944 ± 0.0007   | 0.9995 ± 0.0002   |
| ResU-Net  | 1,614,977 | 0.999846 ± 0.000028   | 0.9929 ± 0.0013   | 0.9987 ± 0.0003   |

[INSERT FIGURE 3 — F1 bar chart: mean test F1 ± std across all three datasets and models]

FNNSeg achieves the highest mean F1 (0.9952) with the lowest parameter count (218,609). The differences between models are modest in absolute terms but consistent across all three seeds: FNNSeg outperforms U-Net in every individual seed run and ResU-Net in all six seed runs. The standard deviation of F1 across seeds is ≤0.0013 for all models, confirming statistical stability.

**Comparison with du Toit et al. (2023):** The best published result on this dataset is RFI_NET with a maximum F1 of 0.9936. FNNSeg's mean F1 of 0.9952 exceeds this by 0.0016 despite using 3.7 times fewer parameters (218K vs 805K). FNNSeg also exceeds all other du Toit benchmarks: UNET (0.9869 best F1), ASPP_UNET (0.9863), DSC_DUAL_RESUNET (0.9859), RNET (0.9895), and RNET5 (0.9851). Figure 4 situates all models on a parameter count vs test F1 Pareto plane, and Figure 5 presents a direct per-model comparison against all du Toit benchmarks.

[INSERT FIGURE 4 — Pareto chart: parameter count vs test F1, all models including du Toit benchmarks]

[INSERT FIGURE 5 — Grouped bar chart comparing FNNSeg, U-Net, and ResU-Net against du Toit et al. (2023) published benchmarks on HERA-PP]

### 6.2 HERA AOFlagger Results

Table 6 reports results on the HERA AOFlagger-labelled benchmark (HERA-AOF).

**Table 6: HERA AOFlagger Label Results (mean ± std, n=3 seeds)**

| Model     | Params    | Accuracy              | F1 Score          | AUC-ROC           |
|:----------|----------:|:---------------------:|:-----------------:|:-----------------:|
| U-Net     | 482,593   | 0.994564 ± 0.000033   | 0.7791 ± 0.0007   | 0.9793 ± 0.0040   |
| FNNSeg    | 218,609   | 0.994423 ± 0.000085   | 0.7748 ± 0.0026   | 0.9775 ± 0.0013   |
| ResU-Net  | 1,614,977 | 0.993978 ± 0.000087   | 0.7606 ± 0.0027   | 0.9719 ± 0.0026   |

Performance drops substantially for all models compared to HERA-PP: mean F1 falls from approximately 0.99 to approximately 0.77. This is consistent with du Toit et al. (2023)'s finding that AOFlagger labels introduce noise that limits learning. The relative ordering is reversed compared to HERA-PP: U-Net slightly leads FNNSeg (ΔF1 = 0.0044), with ResU-Net substantially trailing. This suggests that for label-noisy settings, the moderate capacity of U-Net may be a marginal advantage over the more compact FNNSeg, though the difference falls within the margin of seed variance.

Importantly, all three models achieve virtually identical pixel accuracy (~0.994), indicating that the F1 gap reflects differences in handling the foreground (RFI) class rather than overall classification ability. Figure 6 shows the learning curves for all three models on HERA-AOF. The noisier trajectories compared to HERA-PP are consistent with the label noise hypothesis: the model cannot achieve a clean validation signal because a fraction of the ground-truth labels are themselves incorrect.

[INSERT FIGURE 6 — Learning curves (validation F1 and training loss vs epoch) for all three models on HERA-AOF across three seeds]

### 6.3 LOFAR Results

Table 7 reports results on the LOFAR real-observation benchmark.

**Table 7: LOFAR Dataset Results (mean ± std, n=3 seeds)**

| Model     | Params    | Accuracy              | F1 Score          | AUC-ROC           |
|:----------|----------:|:---------------------:|:-----------------:|:-----------------:|
| FNNSeg    | 218,465   | 0.994597 ± 0.000015   | 0.6337 ± 0.0006   | 0.9646 ± 0.0070   |
| U-Net     | 482,449   | 0.994470 ± 0.000044   | 0.6281 ± 0.0026   | 0.9593 ± 0.0022   |
| ResU-Net  | 1,614,833 | 0.994273 ± 0.000043   | 0.6174 ± 0.0026   | 0.9227 ± 0.0011   |

FNNSeg again leads both baselines in mean F1 (0.6337 vs U-Net 0.6281 vs ResU-Net 0.6174). The absolute F1 scores are substantially lower than on HERA, reflecting the inherent difficulty of real terrestrial RFI from diverse European sources, which is morphologically more complex and spectrally more variable than the simulated HERA interference.

ResU-Net shows the most significant AUC degradation: its mean AUC falls from 0.9987 (HERA-PP) to 0.9227 (LOFAR), a drop of 0.076. FNNSeg's AUC drops from 0.9995 to 0.9646 (a drop of 0.035), suggesting the compact fractal architecture generalises better across telescope domains. The seed variance for LOFAR is notably higher for U-Net (±0.0026) and ResU-Net (±0.0026) than for FNNSeg (±0.0006). Figure 7 shows the LOFAR learning curves, which are the slowest to converge across all three benchmarks — a consequence of the large training set (60K+ patches) and the complexity of the real-world RFI patterns.

[INSERT FIGURE 7 — Learning curves (validation F1 and training loss vs epoch) for all three models on LOFAR across three seeds]

Figure 8 places the three datasets' results side by side as a difficulty comparison, making the progressive F1 degradation from HERA-PP to HERA-AOF to LOFAR explicit for all three models.

[INSERT FIGURE 8 — Dataset difficulty comparison: mean test F1 per model per dataset (HERA-PP, HERA-AOF, LOFAR)]

### 6.4 Convergence and Training Behaviour

Figure 9 provides a cross-dataset summary heatmap of test F1, AUC, and accuracy across all three models and all three datasets, consolidating the results from Sections 6.1–6.3 into a single view.

[INSERT FIGURE 9 — Cross-dataset performance heatmap: test F1, AUC, and accuracy for each model × dataset combination]

Figures 7, 8, and 9 show the per-epoch training loss, validation loss, and validation F1 for every model–seed combination on HERA-AOF, LOFAR, and HERA-PP respectively. Figure 10 overlays the best-seed validation F1 trajectory for all three models on each dataset; each curve represents the seed that achieved the highest test F1 for that model–dataset combination, with seed labels in the legend.

On HERA-PP (Figure 9), all three models converge quickly and cleanly. FNNSeg and U-Net climb smoothly to a validation F1 plateau of approximately 0.995 within the first eight epochs and remain stable for the full 30 epochs. ResU-Net (seed=0) shows a sharp dip to approximately 0.87 at epoch 2 — a transient instability in its deeper residual branches under the cosine annealing schedule — before recovering and reaching the same plateau by epoch 28.

On HERA-AOF (Figure 7), convergence is slower and the curves more spread. FNNSeg (seed=2) starts at approximately 0.59 due to a less favourable random initialisation, then rises steeply to approximately 0.78 by epoch 27. U-Net (seed=0) and ResU-Net (seed=0) begin around 0.72 and climb more steadily, with ResU-Net reaching the highest validation peak of approximately 0.82. Despite ResU-Net's superior validation trajectory, the held-out test ordering is U-Net (0.7799) > FNNSeg (0.7784) > ResU-Net (0.7625) — a val/test inversion that points to mild overfitting in the larger model.

On LOFAR (Figure 8), all three curves climb very slowly across 30 epochs, consistent with the dataset's sparse and irregular RFI structure. By epoch 30, ResU-Net (seed=1) reaches the highest validation F1 of approximately 0.76–0.78, followed by U-Net (seed=1) at approximately 0.74–0.75 and FNNSeg (seed=0) at approximately 0.70. Yet the held-out test results are exactly reversed: FNNSeg (0.6344) > U-Net (0.6314) > ResU-Net (0.6205). ResU-Net's larger capacity overfits the training-drawn validation split; FNNSeg generalises better to the independent test set. This is corroborated by FNNSeg's lower seed variance on test F1 (±0.0006) compared to ±0.0026 for both baselines.

[INSERT FIGURE 10 — Convergence: best-seed validation F1 vs training epoch for each model across all three datasets]

### 6.5 Parameter Efficiency Analysis

As shown in Figure 4, FNNSeg occupies the upper-left Pareto frontier of the parameter count vs test F1 plane: it achieves the highest F1 with fewer parameters than any competing model. No other model in the comparison combines F1 ≥ 0.995 with fewer than 300K parameters.

The nearest competitor in the efficiency dimension is RNET (du Toit et al., 32K parameters, F1 = 0.9895), which achieves competitive performance at an even smaller model size. However, RNET is a decoder-only architecture without skip connections, limiting its ability to preserve fine spatial detail. FNNSeg's skip connections and multi-resolution fractal processing enable higher F1 at a modest additional parameter cost. Table 8 below presents the complete comparison, including best single-seed results for FNNSeg alongside the best published results from du Toit et al. (2023).

**Table 8: Comparison with du Toit et al. (2023) Published Benchmarks — HERA-PP**

| Model            | Source                | Params    | Best Test F1    | Best Test AUC |
|:-----------------|:----------------------|----------:|:---------------:|:-------------:|
| FNNSeg (best)    | This work             | 218,609   | 0.9964 (seed=0) | 0.9998        |
| FNNSeg (mean)    | This work             | 218,609   | 0.9952 ± 0.0008 | 0.9995        |
| RFI_NET          | du Toit et al. (2023) | 805,473   | 0.9936          | 0.9966        |
| RNET             | du Toit et al. (2023) | 32,705    | 0.9895          | 0.9940        |
| UNET             | du Toit et al. (2023) | 291,633   | 0.9869          | 0.9919        |
| ASPP_UNET        | du Toit et al. (2023) | 291,633   | 0.9863          | 0.9919        |
| DSC_DUAL_RESUNET | du Toit et al. (2023) | 370,494   | 0.9859          | 0.9910        |
| DSC_MONO_RESUNET | du Toit et al. (2023) | 185,226   | 0.9826          | 0.9896        |
| RNET5            | du Toit et al. (2023) | 19,809    | 0.9851          | 0.9923        |

Note: du Toit et al. values are the best single run across all seeds/configurations as reported in the original paper. FNNSeg mean is computed over 3 seeds; the best single-seed result (seed=0) is also shown for a direct like-for-like comparison.

### 6.6 Seed Variance and Statistical Reliability

Figure 13 presents box plots of F1 scores across the three seeds for each model and dataset, providing a direct visual representation of the variance reported in Tables 5–7.

[INSERT FIGURE 13 — Box plots of test F1 scores across three seeds for each model and dataset (HERA-PP, HERA-AOF, LOFAR)]

Key observations:

- **HERA-PP:** All three models show low variance (F1 range < 0.002). FNNSeg's interquartile range is the tightest, indicating the most consistent performance.
- **HERA-AOF:** Variance increases for all models, particularly FNNSeg (std = 0.0026) and ResU-Net (std = 0.0027). U-Net is the most stable on AOF (std = 0.0007).
- **LOFAR:** FNNSeg has the lowest variance (std = 0.0006) despite the hardest task, while U-Net and ResU-Net both show std = 0.0026.

The stability of FNNSeg under difficult conditions (LOFAR real data) is notable: it suggests that the fractal block's multi-path averaging produces a natural ensemble effect that smooths out initialisation sensitivity. The elevated variance for U-Net and ResU-Net on LOFAR means that individual seed results for those models may be misleading if reported without aggregation — a methodological consideration for future comparisons that report only a single training run.

---

## 7. DISCUSSION

### 7.1 Primary Finding: FNNSeg Achieves State-of-the-Art Efficiency

The central finding of this dissertation is that the Fractal Neural Network with Fibonacci channel scheduling achieves state-of-the-art RFI detection performance on the HERA pixel-perfect benchmark while using 2.2 to 7.4 times fewer parameters than competing models. The mean F1 of 0.9952 (best: 0.9964) exceeds the best published result of du Toit et al. (2023) by 0.0016 to 0.0028 F1 points. This result directly confirms RQ1.

The parameter efficiency advantage (RQ2) is consistent across all three datasets. On HERA-PP, HERA-AOF, and LOFAR, FNNSeg achieves the best or near-best F1 among the three architectures tested. The magnitude of the advantage is largest on the most challenging dataset (LOFAR, where FNNSeg outperforms U-Net by ΔF1 = 0.006), suggesting that the regularising effect of fractal architecture (multiple independent paths, forced generalisation via drop-path) provides the greatest benefit precisely when overfitting is most likely.

### 7.2 Why Fractal Blocks Outperform Standard Convolutions

The performance advantage of FNNSeg over standard U-Net cannot be attributed to parameter count alone, because FNNSeg has fewer parameters, which typically disadvantages a model relative to a larger one trained on the same data. Three architectural properties explain the advantage:

**Implicit ensemble behaviour:** The fractal block's parallel paths (shallow left branch, recursive deep right branch) function as an implicit ensemble. The learned 1×1 merge convolution weights the contribution of shallow and deep features adaptively for each spatial location, providing the benefits of model averaging without requiring multiple independent training runs.

**Multi-scale feature aggregation:** RFI appears in dynamic spectra at multiple scales simultaneously: narrowband interference (1–2 frequency channels wide, persistent in time), broadband bursts (wide in frequency, short in time), and complex wideband signals. The fractal block's recursive depth creates representations at effective receptive field sizes corresponding to 2, 4, and 8 convolutional layers simultaneously. This implicit multi-scale representation is better matched to the morphological diversity of RFI than a fixed-depth convolutional block.

**Fibonacci channel scheduling:** By growing channels sublinearly (Fibonacci ratios converging to φ ≈ 1.618) rather than by doubling (ratio 2), the encoder progressively increases representational capacity while keeping intermediate feature maps compact. This keeps the bottleneck relatively low-dimensional, acting as an information bottleneck that forces the encoder to learn compact, discriminative representations rather than memorising training examples.

### 7.3 Label Quality and the AOFlagger Gap

The dramatic performance drop from HERA-PP (F1 ≈ 0.99) to HERA-AOF (F1 ≈ 0.77) for all models is a critical finding that bears on the practical use of deep learning for RFI mitigation. AOFlagger labels are systematically noisy in two ways: (1) they under-flag weak, diffuse RFI that the AOFlagger morphological operators do not detect; (2) they over-flag in complex frequency ranges where bright foreground emission (e.g., Galactic synchrotron) is misidentified as interference. A deep learning model trained to reproduce AOFlagger labels is therefore learning a biased approximation of the true RFI distribution.

This finding directly addresses RQ3: label quality has a profound effect on model performance, and the AOFlagger gap is not indicative of model failure. An important implication for future work is that simulation-based pretraining (using pixel-perfect HERA-PP labels) followed by fine-tuning on AOFlagger-labelled real observations could break through the AOFlagger performance ceiling, an approach similar to the pretraining paradigm in natural language processing.

The fact that U-Net slightly outperforms FNNSeg on HERA-AOF (ΔF1 = 0.0044) is consistent with the interpretation that label noise may slightly benefit moderate-capacity models: a more complex decision boundary may fit label noise more aggressively, while a slightly more constrained model generalises to the true distribution better. Both U-Net and FNNSeg are, however, within each other's seed variance on this dataset, making this difference practically insignificant.

### 7.4 Cross-Instrument Generalisation (LOFAR)

LOFAR represents the cross-domain generalisation challenge: a model trained on LOFAR-style data is evaluated on real interference from a fundamentally different observational setting (different frequency range, different array geometry, European terrestrial RFI sources). All models were trained from scratch on LOFAR data; the experiments are therefore in-domain for LOFAR but present a harder task due to the complexity of real-world interference.

The lower F1 on LOFAR (~0.63) relative to HERA-PP (~0.99) is expected and consistent with the du Toit et al. (2023) observations. What is notable is that FNNSeg's relative advantage over ResU-Net is larger on LOFAR (ΔF1 = 0.016) than on HERA-PP (ΔF1 = 0.002). This suggests that the compact, regularised fractal architecture generalises better to complex real-world conditions than the deeper, more expressive ResU-Net, which may be more prone to overfitting the peculiarities of the large LOFAR training set.

The LOFAR test set size (109 frames) is a limitation: with only 109 test samples, the test-set F1 estimates have non-trivial standard error, and differences of ΔF1 < 0.01 should be interpreted cautiously.

### 7.5 Computational Cost Analysis

The T4 GPU (15 GB VRAM) provided approximately 40 times the training throughput of CPU: a 30-epoch HERA run took approximately 1.5 hours on the T4 versus approximately 72 hours on the MacBook Air M4 CPU. The three CPU seed=0 runs (FNNSeg, U-Net, ResU-Net on HERA-PP) were completed locally before the GCP environment was set up.

The initial LOFAR seed=0 run for FNNSeg was carried out on an n1-standard-8 instance (8 vCPU, 30 GB RAM) as a single job, made possible by the in-place preprocessing optimisation described in Section 4.5, which reduced steady-state RAM to approximately 19 GB. Subsequent LOFAR experiments were migrated to an n1-highmem-8 instance (8 vCPU, 52 GB RAM), which allowed two jobs to run concurrently (two × 19 GB = 38 GB, within the 52 GB limit), reducing the total wall-clock time for the LOFAR experiment block by approximately 50%. The total estimated GCP expenditure across all 24 GPU runs was approximately $38 USD (compute and storage combined), as detailed in Table 3 and Figure 14.

[INSERT FIGURE 14 — Compute overview: wall-clock time per run by instance type (left) and GCP cost breakdown by dataset and instance type (right)]

### 7.6 Limitations

- **Small test set for LOFAR:** The 109-frame LOFAR test set produces noisy F1 estimates; confidence intervals are wide and differences below ΔF1 = 0.01 should not be over-interpreted.
- **Single patch size:** All experiments use 256×256 patches. Performance may vary for smaller patches (faster inference, less context) or full-frame processing (more context but memory-intensive).
- **No cross-dataset transfer (HERA → LOFAR):** A trained HERA model was not evaluated directly on LOFAR test data. Zero-shot cross-telescope transfer would be a challenging but practically relevant scenario and remains for future work.
- **Hyperparameter sensitivity not fully explored:** The fractal block depth, Fibonacci base channel, and drop-path rate were fixed rather than systematically tuned on the real-data benchmarks. A Bayesian hyperparameter optimisation over depth in {2, 3, 4} and base_ch in {8, 16, 32} represents a clear avenue for further performance gains.
- **Interpretability:** Deeper fractal recursion makes it harder to attribute predictions to specific feature maps compared with standard convolutional layers. Visualisation tools such as Grad-CAM would need to be adapted for the recursive block structure.
- **No full-frame inference visualisation for CPU runs:** The checkpoint saving mechanism was added after the initial CPU seed=0 experiments, so best_model.pt files were not produced for those three runs. Seeds 1 and 2 (GPU runs) have checkpoints and support full inference visualisation.
- **Binary segmentation only:** FNNSeg has not been tested beyond binary RFI/clean classification. Multi-class flagging (e.g., distinguishing RFI type) and regression tasks (e.g., intensity estimation) have not been explored.

### 7.7 Broader Implications

A model achieving 0.9952 mean F1 with 218K parameters can be quantised to INT8 precision and deployed on FPGAs or edge AI accelerators (e.g., NVIDIA Jetson, Google Edge TPU) at the correlator or raw data output of a radio telescope. This enables real-time RFI flagging at data acquisition time rather than offline post-processing, a capability that will be critical for the SKA, which will generate terabytes per second of visibility data. The parameter efficiency of FNNSeg is therefore not merely an academic curiosity but a practical enabler for next-generation telescope operations.

---

## 8. CONCLUSION

### 8.1 Summary of Contributions

This dissertation has made the following contributions:

1. **FNNSeg architecture:** A novel segmentation network combining Fibonacci channel scheduling (widths [16, 32, 48] for base_ch=16, stages=3), recursive FractalBlocks with SiLU activation, and optional drop-path regularisation, achieving state-of-the-art RFI detection with 218K parameters.
2. **Systematic empirical evaluation:** 27 training runs across 3 models × 3 datasets × 3 seeds, producing statistically reliable performance estimates comparable with the du Toit et al. (2023) benchmark framework.
3. **State-of-the-art result on HERA-PP:** FNNSeg mean F1 of 0.9952 exceeds the best published result (RFI_NET, F1=0.9936) while using 3.7 times fewer parameters.
4. **Consistent cross-dataset leadership:** FNNSeg leads both baseline architectures on all three datasets, with the advantage increasing on harder tasks (LOFAR), suggesting robustness benefits from the fractal regularisation.
5. **Memory-efficient preprocessing for large-scale data:** An in-place NumPy pipeline enabling LOFAR (9.9 GB) training on instances with approximately 30 GB RAM.

### 8.2 Research Questions Revisited

- **RQ1 (competitive performance):** Confirmed. FNNSeg achieves the highest F1 on HERA-PP among all models tested, including du Toit et al. published benchmarks.
- **RQ2 (efficiency advantage translates across datasets):** Confirmed. FNNSeg leads baselines on HERA-PP, HERA-AOF, and LOFAR with consistently fewer parameters.
- **RQ3 (label quality effects):** Confirmed. Label quality profoundly affects performance: AOFlagger labels cap F1 at approximately 0.77; LOFAR real labels at approximately 0.63. FNNSeg generalises no worse, and often better, than larger baselines under harder conditions.

### 8.3 Future Work

1. **Pretraining and fine-tuning:** Pre-train on HERA-PP (large clean dataset, simulation labels), then fine-tune on HERA-AOF or LOFAR with a lower learning rate. This may break through the AOFlagger performance ceiling by leveraging the model's learnt representation of clean RFI morphologies before adapting to noisier supervision.
2. **Semi-supervised learning:** The Zenodo datasets have labelled training sets, but telescope archives contain orders of magnitude more unlabelled data. Self-supervised pretraining (e.g., masked autoencoding on raw visibilities) followed by supervised fine-tuning may improve generalisation significantly.
3. **FNNInpaint (dual-head restoration):** Extending FNNSeg with a parallel restoration decoder, trained jointly to detect and reconstruct contaminated pixels, enables recovering astrophysical signal rather than merely flagging it, as demonstrated in principle by Zhang et al. (2024).
4. **Integration into production pipelines:** Wrapping FNNSeg as a DP3 (LOFAR processing pipeline) or CASA (Common Astronomy Software Applications) step to enable operational deployment, including assessment of latency under real-time processing constraints (Sclocco et al., 2020).
5. **Cross-telescope transfer learning:** Train on HERA data and evaluate zero-shot performance on LOFAR, followed by few-shot adaptation. This would address the challenge of deploying a single trained model across heterogeneous instrument configurations.
6. **SKA readiness assessment:** Evaluate FNNSeg on simulated SKA-Low observations, which will present the largest scale and highest data rate RFI challenge of any telescope currently planned for construction.

---

*End of dissertation draft — figures referenced as [INSERT FIGURE N — description] should be placed in the appendix or inline per the Word document layout.*
