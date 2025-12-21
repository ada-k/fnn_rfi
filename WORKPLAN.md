# FNN_RFI — Workplan (8-week / 2-month Masters timeline)

Project summary
- Objective: design a new Fractal Neural Network (FNN) architecture inspired by fractal and Fibonacci patterns, evaluate it against standard baselines, then adapt/transfer it to Radio Frequency Interference (RFI) detection in radio astronomy using freely available datasets.
- Constraints: all datasets and tools must be freely available; final deliverables include code, dataset links/instructions, proposal, and thesis.

High-level milestones
- Week 0 (preparation): finalize scope, confirm advisor requirements, set up repo, environment, and data storage.
- Weeks 1–2: literature review and baseline selection; collect and validate free RFI datasets; shortlist evaluation metrics.
- Weeks 3–4: design and prototype FNN; implement simple synthetic/benchmark experiments to test properties.
- Weeks 5–6: implement baselines (CNN, RNN/LSTM, Transformer) and train FNN + baselines on RFI task; iterative improvements.
- Weeks 7–8: evaluation, ablation studies, write-up (proposal finalization if needed, thesis chapters, figures), reproducibility packaging, and final submission.

Week-by-week plan
- Week 0 (days 0–3):
  - **Setup**: create virtual environment, CI (basic), repo structure, and starter README.
  - **Deliverable**: working repo skeleton and environment instructions.

- Week 1 (days 4–10):
  - **Literature review**: survey fractal neural nets, Fibonacci-inspired architectures, and prior ML for RFI detection.
  - **Deliverable**: annotated bibliography (1–2 pages) and list of gaps / novelty claims to target.

- Week 2 (days 11–17):
  - **Data**: locate and download freely available radio astronomy data suitable for RFI experiments (see Data sources below). Begin labelling/flagging pipeline or use published labels.
  - **Deliverable**: dataset inventory with download commands and sample preprocess code.

- Week 3 (days 18–24):
  - **Design**: specify FNN topology (fractal blocks, Fibonacci-based scaling, depth/width rules) and initial hyperparameters.
  - **Deliverable**: model spec document and small prototype implementation.

- Week 4 (days 25–31):
  - **Prototype validation**: train FNN on small synthetic datasets or standard vision/audio benchmarks to validate training dynamics.
  - **Deliverable**: short experiment report with training curves.

- Week 5 (days 32–38):
  - **Baselines**: implement and/or adapt baseline models (CNN, RNN/LSTM, Transformer) and prepare unified training pipeline.
  - **Deliverable**: baseline implementations and training scripts.

- Week 6 (days 39–45):
  - **RFI training**: train FNN and baselines on the RFI dataset(s); start hyperparameter tuning.
  - **Deliverable**: initial performance numbers and logs.

- Week 7 (days 46–52):
  - **Evaluation**: run ablations, robustness tests, false-positive analysis, and compute chosen metrics (precision/recall, F1, ROC-AUC, confusion matrices, runtime/complexity).
  - **Deliverable**: consolidated evaluation tables and plots.

- Week 8 (days 53–60):
  - **Write-up & packaging**: finalize proposal/thesis chapters, prepare figures, ensure code reproducibility, create a public release (GitHub) and final backups.
  - **Deliverable**: draft thesis with results, reproducible code, and dataset instructions.

Datasets (sources & notes)
- Use only freely accessible archives and curated datasets. Candidate sources:
  - LOFAR / LOFAR Long-Term Archive (open observation data) — useful for raw spectrograms.
  - VLA / NRAO Science Data Archive — public observations; preprocessing may be required.
  - HERA public data releases — often used in RFI studies and have flagging metadata.
  - ASTRON / AARTFAAC or other public skymap archives.
- Note: If no labeled RFI dataset is available, plan to create labels via a) published flagging outputs (e.g., AOFlagger) or b) semi-automatic labelling + manual verification on small subsets.

Baselines and evaluation
- Baselines: standard CNN (2D conv on spectrograms), RNN/LSTM (sequence modelling), and a lightweight Transformer. Also include a simple heuristic/threshold-based RFI flagger as a baseline.
- Metrics: precision, recall, F1-score, ROC-AUC, per-class accuracy, false positive rate, inference time, model size, and robustness to SNR/artifact variations.

Implementation & reproducibility
- Use PyTorch (recommended) or TensorFlow, with an environment file (`requirements.txt`) and example training scripts.
- Provide clear `run` commands, data-prep notebooks/scripts, and a small test dataset for CI to run quick sanity checks.

Writing deliverables
- Proposal: 3–5 pages including problem statement, related work, proposed FNN innovation, datasets, evaluation plan, and timeline (deliver in Week 2–3 as required by advisor).
- Thesis: structure chapters: Introduction, Related Work, Methods (FNN), Data, Experiments, Results, Discussion, Conclusion, Reproducibility Appendix.

Risks & mitigations
- Risk: no sufficiently large labelled RFI dataset — mitigation: use AOFlagger outputs to generate labels, augment data, or simulate RFI overlays on clean data.
- Risk: FNN doesn't outperform baselines — mitigation: perform ablations, tune capacity, and report negative results with analysis.

Next steps (immediate)
1. Confirm dataset sources you prefer (I can search for specific, labeled RFI datasets and add links).
2. Start the literature review and assemble the annotated bibliography.
3. I'll create `requirements.txt` and a repo skeleton if you want me to proceed.

---
_This workplan is intentionally aggressive for an 8-week target; we can relax scope or extend timelines if needed._
