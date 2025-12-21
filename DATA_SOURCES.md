# Data Sources & Acquisition Notes

Goal
- Provide freely available radio-astronomy data sources and practical notes for obtaining labelled or labelable data for RFI detection experiments.

Primary archives (public)
- LOFAR Long Term Archive (LTA): https://lta.lofar.eu/ — open observations; may require account and bulk download tools.
- NRAO / VLA Science Data Archive: https://archive.nrao.edu/ — public raw data and metadata.
- HERA public data: check HERA data releases on the HERA or related websites (search "HERA public data releases").
- MWA (Murchison Widefield Array): public data available via the MWA archive and ASVO (Australian SKA Pathfinder archive pages).
- AARTFAAC / ASTRON resources: https://www.astron.nl/ — check public datasets and papers.

Helpful tools & software
- AOFlagger (RFI flagging tool): https://github.com/lofar-astron/AOFlagger — use it to generate labels/flags from raw data.
- CASA (Common Astronomy Software Applications) for data handling and imaging: https://casa.nrao.edu/.

Labeling strategies
- Use AOFlagger outputs as initial labels (often used in literature).
- If labeled datasets exist in literature, use those directly; otherwise simulate RFI signals and overlay on clean data to create labelled examples.

Practical notes
- Many archives host large raw datasets; plan for selective downloads (small observation IDs) or using remote processing where possible.
- For initial experiments, convert data to spectrogram/image-like representations (time-frequency) or per-channel time series depending on the model input.

If you want, I can search for explicit public RFI-labeled datasets (paper-backed) and collect direct download links and small sample download commands.
