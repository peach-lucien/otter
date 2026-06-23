# Notebooks

Fifteen interactive walkthroughs. None re-fit the model from scratch, they load pre-computed outputs from `outputs/`. Run `pipeline/04_solve_production.py` + `experiments/anchor_packs/compose_all.py` first if you need to regenerate.

## Reading order

| # | Notebook | What it does | Time |
|---|---|---|---|
| 01 | [`01_quickstart.ipynb`](01_quickstart.ipynb) | Interactive: pick a mouse region, see top-K human partners. Compare strict-π vs packed-π. | 5 min |
| 02 | [`02_trust_map.ipynb`](02_trust_map.ipynb) | Per-parcel multi-source evidence tier. Interactive filter by tier. Which parts of the brain to trust. | 5 min |
| 03 | [`03_anchor_packs.ipynb`](03_anchor_packs.ipynb) | Side-by-side Beauchamp comparison: production vs production+all-packs. Per-region lift table + bar plot. | 5 min |
| 04 | [`04_methodology.ipynb`](04_methodology.ipynb) | Step-by-step FGW: raw FC → costs → M → solver → eval. For people who want to see what's inside. | 15 min |
| 05 | [`05_pagani_2026_validation.ipynb`](05_pagani_2026_validation.ipynb) | HOMER × Pagani 2026 (Nat Neurosci autism subtypes), four-hypothesis arc validating their cross-species claims through HOMER's π. | 10 min |
| 06 | [`06_hodge_2019_layer_markers.ipynb`](06_hodge_2019_layer_markers.ipynb) | HOMER × Hodge 2019 (Nature cortical-layer markers). Beauchamp-independent test of HOMER's anatomical fidelity using cross-species gene expression. | 10 min |
| 07 | [`07_margulies_huntenburg_gradient.ipynb`](07_margulies_huntenburg_gradient.ipynb) | HOMER × Margulies 2016 + Huntenburg 2021, brain-wide principal connectivity gradient. Single global organisational test orthogonal to anchor pairs. | 10 min |
| 08 | [`08_coletta_2020_cross_species_rsn.ipynb`](08_coletta_2020_cross_species_rsn.ipynb) | HOMER × Coletta 2020, cross-species RSN correspondence + ICA-derived data-driven check + network coherence. Stricter version of Pagani Test 1 with the canonical Yeo-7 atlas. | 10 min |
| 09 | [`09_pagani_per_model_translation.ipynb`](09_pagani_per_model_translation.ipynb) | HOMER × Pagani per-mouse-model, exploratory showcase of which human ASD subtype each of 20 mouse autism models translates to via HOMER's π. Limited to subtype-resolution because 1,491-feature decoding wasn't possible. | 10 min |
| 10 | [`10_biccn_cell_type_markers.ipynb`](10_biccn_cell_type_markers.ipynb) | HOMER × BICCN (Yao 2023 + Siletti 2023) cell-type markers. Pvalb, Sst, Th, Gfap, etc. Establishes that HOMER preserves regionally-concentrated cell-type signals (Th, Gfap, Plp1) but not broadly-cortical class markers (interneurons). Side-by-side with Hodge layer markers. | 10 min |
| 11 | [`11_enigma_cross_disorder.ipynb`](11_enigma_cross_disorder.ipynb) | HOMER × ENIGMA cross-disorder spatial validation. Phase 1 (in-notebook): per-disorder predictions at parcel resolution show HOMER's cross-disorder correlations at r > 0.97, confirming no disorder-specificity. Phase 2 (needs external ENIGMA Toolbox download): comparison against published Cohen's d maps. | 10 min |
| 12 | [`12_fulcher_2019_multimodal_gradient.ipynb`](12_fulcher_2019_multimodal_gradient.ipynb) | HOMER × Fulcher 2019 (PNAS multimodal mouse-cortex gradients), routes mouse T1w:T2w + cytoarchitecture through π; both reproduce the human HCP myelin map. Structural, Beauchamp-independent, 3-panel figure. | 5 min |
| 13 | [`13_balsters_2020_mfc_divergence.ipynb`](13_balsters_2020_mfc_divergence.ipynb) | HOMER × Balsters 2020 (PNAS), falsification test: mouse medial frontal cortex routes to human medial/premotor/cingulate cortex and avoids dlPFC, as the connectivity evidence predicts. | 5 min |
| 14 | [`14_transbrain_2025_benchmark.ipynb`](14_transbrain_2025_benchmark.ipynb) | HOMER × TransBrain 2025 (Nature Methods), head-to-head against a sibling mouse↔human translation method, plus scoring HOMER on TransBrain's literature homology benchmark. | 5 min |
| 15 | [`15_buckner_krienen_2013_tethering.ipynb`](15_buckner_krienen_2013_tethering.ipynb) | HOMER × Buckner & Krienen 2013 (TICS), negative control: HOMER's π is dramatically sparsest over human association cortex, as the tethering hypothesis predicts. | 5 min |

Notebooks 05-15 are **third-party showcases**, each takes a published cross-species paper and tests whether HOMER's π reproduces (or refines) its findings. The template (load HOMER, load paper's source data, route through π, compare, interpret) generalises to any future paper with usable cross-species supplementary data.

## Required widgets

Notebooks 01 and 02 use `ipywidgets` for interactivity. If you see a static-looking output, install:

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## Archive

`archive/` keeps two older notebooks from the iteration period:
- `02_explore_results.ipynb`, the historical "comprehensive comparison" notebook (superseded by `03_anchor_packs.ipynb` + `02_trust_map.ipynb`).
- `03_compare_model_levels.ipynb`, fits all four model levels side-by-side (UnsupervisedGW / SupervisedFGW / MultimodalFGW / HierarchicalFGW). Re-fits from scratch (~5 min), so it's slow but instructive.
