````markdown
# Progress Log

This file tracks actions performed and next steps for the `fnn_rfi` project.

Completed (local repo changes)
- Added `WORKPLAN.md` — overall 8-week plan and milestones.
- Added `FNN_SPEC.md` — FNN design notes and experiment ideas.
- Added `DATA_SOURCES.md` — public radio-astronomy archives and labeling notes.
- Added `PROPOSAL.md` and `THESIS_DRAFT.md` — skeletons for proposal and thesis.
- Implemented a minimal PyTorch prototype:
  - `models/fnn.py` — FractalBlock, ConvBlock, `FNN` class (Fibonacci-driven widths)
  - `models/__init__.py`
  - `scripts/train_toy.py` — small CIFAR10 smoke-test runner
  - `requirements.txt`
- Added `LIT_REVIEW.md` and `LIT_SUMMARIES.md` for literature tracking.

Environment & running notes
- You indicated you're on Arch Linux on an HP laptop without a GPU.
- Earlier run attempt failed with: ModuleNotFoundError: No module named 'torch'.
- Recommended local run steps (using a Python venv named `.venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r fnn_rfi/requirements.txt
python3 fnn_rfi/scripts/train_toy.py --batch 64 --subset 256
```

If your venv is elsewhere, replace `.venv/bin/python3` accordingly.

Next actions (in progress / planned)
- Fetch and summarise ~20 key papers (FractalNet variants, Fibonacci architecture usage, RFI ML papers).
- Implement baseline models (CNN, RNN/LSTM, Transformer) with matching parameter budgets.
- Run synthetic experiments and validate training stability on CIFAR10 and small spectrogram data.
- Collect labeled RFI datasets and prepare data-prep scripts.

Notes on reproducibility
- All code includes docstrings and explanatory comments. Keep this file updated with runs and seed values.

Recent run (smoke test)
- Environment: `.venv` Python used when running the script on 2025-12-21.
- Command: `python3 fnn_rfi/scripts/train_toy.py --batch 64 --subset 256`
- Result: CIFAR10 subset training ran successfully (one epoch smoke test). Sample output:
  - `Model params: 17856`
  - `batch 0 loss 2.3172`
  - `avg loss: 2.2989`

````
