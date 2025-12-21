# Data layout and download commands

This file documents where data and papers live, what is already downloaded, and commands to fetch common resources.

Current local data (repo root `data/`)
- 1605.07648v4.pdf  (FractalNet paper)
- 2001.03389v2.pdf  (Realtime RFI mitigation / Apertif)
- 2005.08992v2.pdf  (Deep learning improves identification of RFI)
- cifar-10-batches-py/ and cifar-10-python.tar.gz (CIFAR10 downloaded by torchvision)

After running `scripts/organize_data.py`, these will be moved under `fnn_rfi/data/` in subfolders:
- `fnn_rfi/data/papers/` — collected PDFs
- `fnn_rfi/data/cifar/` — CIFAR files

Useful links and commands

 - AOFlagger (RFI flagging tool; can generate labels):
   - Overview: AOFlagger is a mature, widely used tool for automated radio-frequency interference detection and flagging maintained by ASTRON. It inspects time-frequency data (e.g., measurement sets) and produces flag masks that can be converted into training labels for ML workflows.
   - Official source: https://gitlab.astron.nl/AOFlagger/AOFlagger
   - Notes on mirrors: there are community mirrors (e.g., GitHub) but the authoritative repo and docs live on the ASTRON GitLab. If a GitHub clone fails or is outdated, prefer the GitLab repo above.
   - Typical install options:
     - Use prebuilt system packages if your distribution provides them (uncommon).
     - Recommended: clone the GitLab repo and build with CMake. You will likely need system build deps (cmake, build-essential, libfftw3-dev, boost, etc.). Example (may require adapting to your distro):

```bash
git clone https://gitlab.astron.nl/AOFlagger/AOFlagger.git
cd AOFlagger
mkdir build && cd build
cmake ..
make -j$(nproc)
# optional: sudo make install
```

   - Running AOFlagger at a high level:
     - AOFlagger typically accepts radio-astronomy data (e.g., measurement sets) and produces flag masks or flagged output. Invocation details depend on the build/version; consult the repo `README` for exact CLI flags.
     - A minimal example (replace with your paths and check installed binary name):

```bash
# example; your binary may be named `aoflagger` or similar
aoflagger -i /path/to/input.ms -o /path/to/output.mask
```

   - How we'll use AOFlagger outputs in this repo:
     - If you produce AOFlagger masks (per time-frequency pixel), place them under `fnn_rfi/data/labels/` with filenames matching the spectrograms or measurement set names.
     - I will add a small wrapper script (`scripts/aoflagger_wrapper.py`) that converts AOFlagger outputs into mask arrays compatible with `SyntheticSpectrogramDataset` and `UNet` training.

   - If you want me to try building AOFlagger here, I can attempt it — but it may require installing system packages (build tools and libs). Tell me if you want me to try and I will proceed and log any missing dependencies.

- LOFAR Long Term Archive (LTA) — public LOFAR observations (may require eduID/account to download):
  - https://lta.lofar.eu/

- NRAO / VLA Science Data Archive — public radio-astronomy data:
  - https://archive.nrao.edu/

- HERA public data releases (search project pages / data portals):
  - https://reionization.org/data/

- MWA / ASVO archives (public observations):
  - https://www.mwatelescope.org/ (project pages) and ASVO portals for direct queries.

Notes about downloads
- Many radio-astronomy archives require account registration and selection of specific Observation IDs. If you provide an ObsID I can add a precise `wget`/`curl` command.
- If no labeled RFI dataset is available, plan to create labels via AOFlagger or simulate RFI overlays on clean data.

Example: prepare a folder for processing (after you obtain/produce observation files)
```bash
mkdir -p fnn_rfi/data/raw_observations fnn_rfi/data/labels
# place your observation files into fnn_rfi/data/raw_observations
# place AOFlagger-produced masks into fnn_rfi/data/labels (matching filenames)
```

If you want, run the organiser script now to move existing files into `fnn_rfi/data/`:
```bash
python3 scripts/organize_data.py
```
