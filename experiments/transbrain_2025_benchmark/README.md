# TransBrain comparison

This directory compares OTTER with
[TransBrain](https://doi.org/10.1038/s41592-025-02961-3), a published
mouse-to-human phenotype-translation framework based on graph embeddings and
dual regression.

The analyses place both methods on common Brainnetome-region representations
and examine literature-curated region pairs, a functional-connectivity
gradient, a Magel2 mutation pattern and an anterior-insula optogenetic map.
Additional scripts export regional distributions, localization summaries and
round-trip results.

This is a common-ground comparison, not a superiority test. OTTER and
TransBrain differ in their inputs, spatial resolution and output
representation; several benchmark territories also overlap OTTER's curated
regional supervision. Method-specific point estimates are therefore reported
descriptively unless a direct statistical comparison is implemented.

## Scripts

| File | Purpose |
|---|---|
| `01_transbrain_benchmark.py` | Runs the regional homology and shared-phenotype comparisons. |
| `02_plot.py` | Plots the primary comparison. |
| `03_transbrain_advanced.py` | Computes regional agreement and phenotype-map summaries. |
| `05_aiopto_headtohead.py` | Translates the anterior-insula optogenetic map with both methods. |
| `06_bn_distributions.py` | Exports Brainnetome-region distributions for benchmark regions. |
| `07_benchmark_summary.py` | Summarizes the exported regional distributions. |
| `08_roundtrip_maps.py` | Computes mouse-to-human-to-mouse round-trip summaries. |
| `09_localization_distributions.py` | Exports localization distributions for selected regions. |

## Inputs

Install the third-party TransBrain package:

```bash
pip install transbrain
```

The scripts use `data_external/transbrain_2025/`, the canonical OTTER coupling,
cached parcellations and the principal-gradient output. Gitignored inputs are
provided in the Zenodo reproduce bundle. `07_benchmark_summary.py` reads
`outputs/logs/transbrain_bn_distributions.json`, so run script `06` first when
regenerating that file.

## Run

From the repository root:

```bash
PYTHONPATH=src python experiments/transbrain_2025_benchmark/01_transbrain_benchmark.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/02_plot.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/03_transbrain_advanced.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/05_aiopto_headtohead.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/06_bn_distributions.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/07_benchmark_summary.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/08_roundtrip_maps.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/09_localization_distributions.py
```

Outputs are written under `outputs/logs/` and `outputs/figures/`; each script's
module documentation names its specific products.
