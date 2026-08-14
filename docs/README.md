# OTTER Documentation

Index of the documentation set.

## Top-level docs

1. **[01_overview.md](01_overview.md)**. Scope, intended users, headline number
2. **[02_methods.md](02_methods.md)**. Fused Gromov-Wasserstein formulation, the anchor mechanism, hyperparameters
3. **[03_results.md](03_results.md)**. The six results, the trust map, the properties that transfer through π
4. **[04_anchor_packs.md](04_anchor_packs.md)**. The anchor packs, citations, composition recipe, pid registry
5. **[05_limitations.md](05_limitations.md)**. Constraints on interpretation
6. **[06_extending.md](06_extending.md)**. Adding a new modality / anchor pack / species
7. **[07_pipeline.md](07_pipeline.md)**. End-to-end reproduction recipe

## Recommended reading paths

- Reviewer or first-time reader: 01 → 03 → 05 (overview, results, limitations). 02 and 04 carry the methods detail.
- Downstream user querying π: 01 → 03 → 04 (which π and which packs to use).
- Method extender: 02 → 04 → 06 (methods, current packs, extension recipe).
- Researcher reproducing: 07 → 02 → 03 (pipeline, methods, results).

## Interactive

- **[index.html](index.html)**. OTTER Mapping Explorer. Static, self-contained 3D viewer for mouse↔human couplings. Search a parcel, see top-K partners, toggle mouse-shell or human-surface overlays. No backend or Python install is needed. When published, this is what GitHub Pages serves at the repo URL.

## See also

- **[Top-level README](../README.md)**. Install + quickstart
- **[notebooks/](../notebooks/)**. Interactive walkthroughs (load π, click regions, see top-K)
- **[pipeline/README.md](../pipeline/README.md)**. Reproduction script map
- **[experiments/README.md](../experiments/README.md)**. Per-pack and ablation runners
- **[outputs/README.md](../outputs/README.md)**. Generated artefact catalogue
