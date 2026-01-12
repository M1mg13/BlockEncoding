BlockEncoding — Snapshot README
================================

This archive contains the BlockEncoding implementation and the experiment driver used for the paper.
It is a snapshot intended for reviewers and archival distribution.

Contents.
- BlockEncoding.py : core implementation of per-site Unitary‑SVD, dilation, TN encoders and helpers.
- experiments.py   : experiment driver that generates per-instance JSON results and plots.
- Images/          : figure PNGs produced by the driver (six figures used in the paper).
- results_full/    : per-instance JSON outputs produced by the driver (may be included or omitted).

Requirements.
- Python 3.8 or newer.
- Minimal Python packages: numpy, matplotlib, seaborn, pandas.
- Optional: scipy if your environment uses it for SVD backends (not required if numpy.SVD is fine).

Quick usage.
- To recompute JSON results (heavy).

  python3 experiments.py --mode results --repeat 1 --allowed-mem-gb 30
- To regenerate figures from existing JSON results (light).

  python3 experiments.py --mode plots
- To run both (results + plots).

  python3 experiments.py --mode all
- To lower memory pressure set a smaller allowed memory.

  python3 experiments.py --mode results --allowed-mem-gb 8

Notes on modes and memory.
- Mode "results" runs the SVD-heavy experiments and writes JSON files to results_full/.
- Mode "plots" only reads results_full/ and writes PNGs to Images/.
- Default SVD memory guard is ~30 GiB (configurable with --allowed-mem-gb).
- The driver will skip instances whose SVD footprint estimate exceeds the guard and write a skipped JSON.

Produced artifacts.
- results_full/summary_full.csv contains a compact table of all runs.
- Each per-instance JSON contains per-site svals, per-site betas, gamma_log and ancilla estimates.
- Images/ contains the six PNG figures referenced in the paper.

Reproducibility guidance.
- Use --mode plots to avoid recomputing heavy SVDs when you only need figures.
- To reproduce a single small example, use --repeat 1 and a small instance (the default grid and QUBO sizes are modest).
- If you run full experiments, expect runtime and memory to scale with chosen sizes; use the allowed-mem-gb flag to protect your machine.

License and citation.
- This snapshot does not include a formal license file.
- For distribution, attach a permissive license (MIT or Apache‑2.0 recommended).
- Please cite the accompanying paper when using this code or the reported experiments.

Contact.
- The code is provided as a snapshot for review.
- For further requests (code updates, repo management, larger result sets) contact the author listed on the paper.

