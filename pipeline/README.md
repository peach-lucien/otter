# Pipeline

End-to-end reproduction recipe. Run these in order to recreate every artefact in `outputs/`.

## Quick start

```bash
# Set up env
conda env create -f env.yml && conda activate otter
pip install -e ".[dev]"

# Run the full pipeline (in order)
PYTHONPATH=src python pipeline/02_build_anndata.py
PYTHONPATH=src python pipeline/03_build_costs.py
PYTHONPATH=src python pipeline/04_solve_production.py
PYTHONPATH=src python pipeline/05_evaluate.py        # orchestrator for 05a-05j
PYTHONPATH=src python pipeline/05g_compute_trust.py
PYTHONPATH=src python pipeline/06_bootstrap.py
PYTHONPATH=src python pipeline/07_build_artefacts.py
PYTHONPATH=src python pipeline/08_build_gui.py
```

## Scripts

| Script | Purpose | Outputs |
|---|---|---|
| `00_external/` | External data downloads (Allen, Domhof, Knox, Beauchamp) | `data_external/` |
| `02_build_anndata.py` | Build mouse + human AnnData caches | `outputs/anndata/*.h5ad` |
| `03_build_costs.py` | Precompute all FC + SC + xyz + gene cost matrices | `outputs/anndata/full_costs.npz` |
| `04_solve_production.py` | Fit the fc_plus_SC point-anchor π | `outputs/coupling/pi_fc_plus_SC.npy` |
| `05_evaluate.py` | **Orchestrator**, runs the substeps below in order | |
| `05g_compute_trust.py` | Per-parcel multi-source trust map | `outputs/coupling/trust_score_*.npz` |
| `06_bootstrap.py` | 40-iter subject-level bootstrap stability | `outputs/coupling/bootstrap_*.npz` |
| `07_build_artefacts.py` | Comparison table + figures + interactive viewer | `outputs/comparison/`, `outputs/figures/`, viewer HTML |
| `08_build_gui.py` | Region-first mapping GUI with model selector, trust filters, and top-K partner summaries | `outputs/gui/index.html`, `outputs/gui/gui_data.json` |

## Component evaluation scripts (called by 05_evaluate.py)

These are run automatically by `05_evaluate.py`. They can also be invoked individually for partial re-runs.

| Script | Evaluation type |
|---|---|
| `05a_anchor_cv.py` | Held-out anchor CV (leave-one-network-out) across all configs |
| `05b_fc_translation.py` | FC translation Pearson r (in-sample + subject-CV held-out) |
| `05c_null_distributions.py` | `random_pi` + `permuted_anchor` null trials |
| `05d_full_space_eval.py` | Full-space top-K + mean rank over all 2094 human parcels |
| `05f_beauchamp_validation.py` | External validation against Beauchamp 2022's 22 pairs |
| `05h_region_anchor_cv.py` | Held-out region-anchor CV (validates region-anchor mechanism) |
| `05j_region_level_eval.py` | Region-level top-K (Beauchamp-22 + JuBrain candidate sets) |
| `05e_knox_vs_standard_sc.py` | Comparative: Knox 2019 voxel SC vs default summary SC |

## Fitting with anchor packs (after running 04_solve_production)

The fc_plus_SC π uses only the 21 Garin point anchors. To fit the **with-packs π without the anchor warp**, described in `docs/04_anchor_packs.md`, run:

```bash
PYTHONPATH=src python experiments/anchor_packs/compose_all.py
```

`pi_canonical.npy` adds the anchor-warped spatial cost to that fit and is what `load_pi()` returns. The individual pack runners in `experiments/anchor_packs/` produce per-pack variants for ablation and inspection.

## Skipping slow steps

The null distribution step can take ~30 min. To skip:

```bash
python pipeline/05_evaluate.py --skip 05c_null_distributions.py
```

## Path conventions

All scripts expect `PYTHONPATH=src` and operate relative to repository root. They are idempotent (safe to re-run; cached cells are skipped) unless invoked with `--recompute`.
