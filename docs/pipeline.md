# End-to-end replication pipeline

How to reproduce the headline results from a clean checkout. Every step is
idempotent and saves to `outputs/`.

## 0. Environment

```bash
conda env create -f env.yml && conda activate homer
pip install -e ".[dev]"
pytest -q                      # 83 tests, ~10 s
```

## 1. External data download (slow, network-dependent)

```bash
PYTHONPATH=src python pipeline/00_external/00_inspect_masks.py
PYTHONPATH=src python pipeline/00_external/00b_verify_alignment.py
PYTHONPATH=src python pipeline/00_external/00c_align_mouse_to_ccf.py
PYTHONPATH=src python pipeline/00_external/01_mouse_sc.py
PYTHONPATH=src python pipeline/00_external/02b_mouse_genes_direct.py    # ~1h, only ~25% of Allen ISH datasets have grid data
PYTHONPATH=src python pipeline/00_external/03_human_sc.py
PYTHONPATH=src python pipeline/00_external/04_human_genes.py
PYTHONPATH=src python pipeline/00_external/05_orthologs.py
```

You'll need `~3 GB` of disk in `data_external/`. See
[`pipeline/00_external/README.md`](../pipeline/00_external/README.md) for
details on each dataset.

## 2. Build per-species AnnData

```bash
PYTHONPATH=src python pipeline/02_build_anndata.py
```

Reads the colleague's two `corrs_*.mat` files, produces:
- `outputs/anndata/mouse.h5ad` (1864 nodes × 105 subjects)
- `outputs/anndata/human.h5ad` (2094 nodes × 113 subjects)

Each h5ad's `uns` carries the mean FC matrix (`fc_mean`), per-cell observation
count (`fc_n_obs`), and the parsed `var` table with anchor labels.

## 3. Build cost matrices

```bash
PYTHONPATH=src python pipeline/03_build_costs.py        # orchestrator (runs 03a, 03b, 03c)
```

Or run individually:
- `03a_build_full_costs.py` → FC + xyz costs in `outputs/anndata/full_costs.npz`
- `03b_build_spatial_costs.py` → adds the per-species xyz GW cost
- `03c_build_multimodal_costs.py` → adds SC, gene-coexpression, M_gene, M_anchor

## 4. Solve the production model

```bash
PYTHONPATH=src python pipeline/04_solve_production.py                # MultimodalFGW(use_sc=True)
PYTHONPATH=src python pipeline/04_solve_production.py --config fc_only   # SupervisedFGW
PYTHONPATH=src python pipeline/04_solve_production.py --multistart       # 5-init multistart sanity
```

Saves:
- `outputs/coupling/pi_fc_plus_SC.npy` — production coupling (1864 × 2094)
- `outputs/coupling/pi_fc_plus_SC.json` — config + fit info sidecar

## 5. Evaluate

```bash
PYTHONPATH=src python pipeline/05_evaluate.py            # orchestrator (runs all 3 substeps)
```

Or individually:
- `05a_anchor_cv.py` — leave-one-network-out CV across 13 configs (~5 min, resumable)
- `05b_fc_translation.py` — FC-translation Pearson r per production config
- `05c_null_distributions.py` — random_pi (50 trials) + permuted_anchors (5 trials) per network
- `05d_full_space_eval.py` — full-space (n_h=2094) recovery for production configs
- `05e_knox_vs_standard_sc.py` — comparative: Knox leaf-level vs Allen summary-structure SC LONO
- `05f_beauchamp_validation.py` — external validation against Beauchamp 2022's 22 mouse↔human pairs

Outputs land in `outputs/logs/*.json`. Already-cached cells are skipped;
pass `--recompute` to force a full recompute.

## 6. Bootstrap stability

```bash
PYTHONPATH=src python pipeline/06_bootstrap.py --config fc_plus_SC --n-iter 40   # ~10 min
```

Saves `outputs/coupling/bootstrap_aggregate_fc_plus_SC.npz` with per-cell mean
+ std of π across 40 subject-bootstrap resamples, plus stability summary in
`outputs/logs/bootstrap_summary_fc_plus_SC.json`. Pass `--config fc_only` to
also get the FC-only baseline (saves to the corresponding `_fc_only` files).
A legacy `bootstrap_summary_fc_only_legacy.json` from a pre-fix FC-only run
(`bootstrap_summary.json`) is kept for reference but the per-config files are
authoritative.

## 7. Build artefacts

```bash
PYTHONPATH=src python pipeline/07_build_artefacts.py     # comparison table + figures
PYTHONPATH=src python pipeline/07b_build_viewer.py       # interactive 3D viewer
```

Produces:
- `outputs/comparison/comprehensive_table.csv` — wide table of all configs
- `outputs/comparison/per_network_top1.csv` — long form, configs × networks
- `outputs/comparison/comparison_summary.md` — markdown summary
- `outputs/figures/13_comprehensive_comparison.png` — 4-panel headline bars
- `outputs/figures/14_config_x_network_heatmap.png` — full heatmap
- `outputs/viewer/index.html` — self-contained interactive viewer

## Headline numbers you should see

After running steps 1–7 end-to-end on the original cohort:

| Metric                                | Value           |
|---------------------------------------|-----------------|
| Anchor CV top-1 (production)          | **81%**         |
| Anchor CV top-5                       | **100%**        |
| FC translation Pearson r (overall)    | **0.36**        |
| FC translation r (within-network)     | **0.45**        |
| Null z-score vs random π              | **+7.5**        |
| Null z-score vs permuted-anchor       | **+17.8**       |
| Subject-CV gap (test − train)         | **−0.04 ± 0.01** |
| Bootstrap mean stability              | **0.98**        |

See [`docs/results.md`](results.md) for the full per-config breakdown.

## Time budget

| Step | Wallclock |
|------|-----------|
| 1. External data download | ~3 hours (network-bound, mostly Allen ISH) |
| 2. Build AnnData          | ~1 minute |
| 3. Build costs            | ~2 minutes |
| 4. Solve production       | ~15 seconds |
| 5. Evaluate               | ~5 minutes (CV) + ~10 minutes (nulls) |
| 6. Bootstrap (40 iter)    | ~10 minutes |
| 7. Build artefacts        | ~30 seconds |
| **Total (excl. download)** | **~25 minutes** |
