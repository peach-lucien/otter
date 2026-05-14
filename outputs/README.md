# Outputs

All generated artefacts. Files here are recreated by the pipeline (see `pipeline/README.md`) — nothing in this directory is hand-edited.

## Layout

```
outputs/
├── anndata/          # mouse.h5ad, human.h5ad, full_costs.npz  (input to 04+)
├── coupling/         # All π files + bootstrap + trust maps
├── logs/             # All evaluation JSONs (Beauchamp, region-level, anchor-CV, …)
├── comparison/       # comprehensive_table.csv + per-network heatmap CSV
├── figures/          # PNG/HTML figures from pipeline/07
├── viewer/           # Lightweight 3D coupling viewer (HTML)
└── gui/              # Region-first mapping explorer (HTML + JSON sidecar)
```

## π files (`coupling/`)

The π matrix (1864 × 2094) is HOMER's core output. We ship several variants:

| File | What | When to use |
|---|---|---|
| `pi_fc_plus_SC.npy` | **Production point-anchor π** (21 Garin anchors only, no packs) | Strictest validation baseline; what most internal CV uses |
| `pi_fc_plus_SC_with_all_packs.npy` | **Recommended π** — production + 5 default anchor packs | **Use this for downstream queries** |
| `pi_fc_plus_SC_with_atlas_regions.npy` | Production + 15 atlas-derived Garin region anchors | Ablation; superseded by per-pack |
| `pi_fc_plus_SC_with_soft_atlas_regions.npy` | Same as above but with soft anchors (lam_outside=0.15) | Justification for the soft default |
| `pi_fc_plus_SC_with_M1.npy` | Production + supplementary M1 point anchor | Historical (EXP-1, hippocampal supp) |
| `pi_fc_plus_SC_with_M1_hippo.npy` | Production + M1 + 4 hippocampal point anchors | Earlier hippocampal coverage experiment |
| `pi_fc_plus_SC_with_biccn_motor.npy` | Production + BICCN motor pack only (pids 30, 31) | Per-pack ablation |
| `pi_fc_plus_SC_with_tectum.npy` | Production + tectum pack only (pids 32, 33) | Per-pack ablation |
| `pi_fc_plus_SC_with_olfactory.npy` | Production + olfactory pack only (pids 34, 35) | Per-pack ablation |
| `pi_fc_plus_SC_with_cingulate.npy` | Production + cingulate pack only (pids 36, 37) | Per-pack ablation (opt-in pack) |
| `pi_fc_plus_SC_with_amygdala.npy` | Production + amygdala pack only (pid 38) | Per-pack ablation |
| `pi_fc_plus_SC_with_hippocampal.npy` | Production + hippocampal pack only (pids 39-42) | Per-pack ablation |
| `pi_fc_plus_SC_with_lateral_pfc.npy` | Production + lateral PFC pack only (pids 45, 46) | Per-pack ablation (opt-in dlPFC) |
| `pi_fc_plus_SC_per_region_xyz_*.npy` | TOPO-1 ablation outputs | Convergent negative; see archive iteration log |
| `pi_fc_plus_SC_xyz_zero.npy` | Production with xyz=0 globally | TOPO-1 control |
| `pi_quickstart.npy` | Demo π for `notebooks/01_quickstart.ipynb` | Notebook only |

## Bootstrap (`coupling/bootstrap_*.npz`)

40 subject-bootstrap iterations on the production π — used by the trust map (`per_row_stability`) and for confidence intervals.

## Trust maps (`coupling/trust_*.npz`)

| File | What |
|---|---|
| `trust_score_fc_plus_SC.npz` | 3-signal internal trust (bootstrap + concentration + FC similarity) on production π |
| `trust_score_fc_plus_SC_with_M1_hippo.npz` | Same on the supplementary-anchor π |
| `trust_multisource_all_packs.npz` | **5-tier multi-source trust** on production-with-packs π — the recommended map |

## Logs (`logs/`)

JSON outputs from every evaluation script:

- `beauchamp_validation*.json` — Beauchamp 2022 external validation results (one per π variant)
- `region_level_eval*.json` — Region-level top-K with Beauchamp-22 or JuBrain candidate sets
- `anchor_cv_*.json` — Held-out anchor CV results
- `fc_translation.json` — FC translation Pearson r
- `bootstrap_summary_*.json` — Bootstrap aggregate stats
- `null_*.json` — Null distribution results
- Most file names indicate the π file they were computed on.

## Comparison + figures + viewer + GUI

Built by `pipeline/07_build_artefacts.py`. Standalone HTML viewer at `viewer/index.html` is the most useful exploration surface — click a mouse parcel, see top-K human partners + trust tier.

For the newer region-first explorer, run:

```bash
PYTHONPATH=src python pipeline/08_build_gui.py
```

This writes `outputs/gui/index.html` and `outputs/gui/gui_data.json`. It compares the baseline Garin-anchor model against the recommended all-packs model when both π files are present.
