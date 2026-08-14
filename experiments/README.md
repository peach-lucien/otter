# Experiments

Reproducible one-off experiments. The `src/otter/` library is the production code; every file here is a research script that produced a specific result documented in `docs/`.

## Layout

```
experiments/
├── anchor_packs/                   # Per-pack experiment runners (default + opt-in)
├── ablations/                      # Methodology ablations (soft anchors, marginals, xyz)
├── autism_subtypes/                # Pagani 2026 4-hypothesis arc + ABIDE + gene-set tests
├── hodge_2019_cortical_layers/     # Hodge 2019 cortical-layer marker translation
├── margulies_2016_principal_gradient/  # Margulies 2016 + Huntenburg 2021 brain-wide gradient
├── fulcher_2019_multimodal_gradient/   # Fulcher 2019 multimodal mouse-cortex hierarchy → human myelin
├── schaeffer_2020_mfc_divergence/  # Schaeffer et al. 2020 frontal-cortex falsification test
├── transbrain_2025_benchmark/      # TransBrain 2025 sibling-method head-to-head
├── buckner_krienen_2013_tethering/ # Buckner & Krienen 2013 tethering negative control
├── coletta_2020_cross_species_rsn/ # Coletta 2020 cross-species RSN correspondence
├── biccn_2023_cell_types/          # BICCN (Yao 2023 + Siletti 2023) cell-type markers
├── enigma_cross_disorder/          # ENIGMA cross-disorder spatial validation
├── whitesell_2021_dmn/             # Whitesell 2021 DMN refinement note
├── pagani_2026_per_model/          # Per-mouse-model exploratory translation
└── outputs/                        # Cached intermediate results from these runs
```

Each subdirectory has its own `README.md` documenting its scripts, inputs, and outputs.

## anchor_packs/, region-anchor experiments

12 per-pack runners that fit production-with-pack π and report Beauchamp + region-level deltas. `compose_all.py` produces the pre-warp `pi_fc_plus_SC_with_all_packs.npy`, retired and kept only to reproduce published comparisons; the canonical coupling adds the anchor-warped spatial cost and is what `load_pi()` returns. See [`anchor_packs/README.md`](anchor_packs/README.md) for the per-pack table and citations.

## ablations/, methodology ablations

Three ablations that justify production design choices (soft anchors, uniform marginal, area-level xyz weighting). In all three the production defaults outperform the variants. See [`ablations/README.md`](ablations/README.md).

## Third-party validations

Each `<paper>_*/` subdirectory takes a published cross-species paper and tests whether OTTER's π reproduces or refines its findings. Outputs are consumed by the notebooks in `notebooks/` and are the source of the numbers in `docs/03_results.md`.

