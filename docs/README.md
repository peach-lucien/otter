# HOMER Documentation

Navigation hub. Read in order if new to the project; jump around once oriented.

## Top-level docs

1. **[01_overview.md](01_overview.md)**. What HOMER does, who it's for, the headline number
2. **[02_methods.md](02_methods.md)**. Fused Gromov-Wasserstein formulation, the anchor mechanism, hyperparameters
3. **[03_results.md](03_results.md)**. Headline numbers, per-region trust map, what works and what doesn't
4. **[04_anchor_packs.md](04_anchor_packs.md)**. The anchor packs, citations, composition recipe, pid registry
5. **[05_limitations.md](05_limitations.md)**. What HOMER can't tell you
6. **[06_extending.md](06_extending.md)**. Adding a new modality / anchor pack / species
7. **[07_pipeline.md](07_pipeline.md)**. End-to-end reproduction recipe

## Recommended reading paths

- **Reviewer / first-time reader**: 01 → 03 → 05 (overview, results, limitations). Skip 02 and 04 unless you want methods detail.
- **Downstream user planning to query π**: 01 → 03 → 04 (figure out which π and which packs to use).
- **Method extender**: 02 → 04 → 06 (methods, current packs, how to add).
- **Researcher reproducing**: 07 → 02 → 03 (pipeline, methods, results).

## Interactive

- **[index.html](index.html)**. HOMER Mapping Explorer. Static, self-contained 3D viewer for mouse↔human couplings: search a parcel, see top-K partners, toggle mouse-shell / human-surface overlays. No backend, no Python install needed. When published, this is what GitHub Pages serves at the repo URL.

## Archive

`archive/` preserves the full iteration history:

- **[iteration_log.md](archive/iteration_log.md)**. The 22-section research log of HOMER's development (originally `results.md`). Full provenance for every claim in the top-level docs.
- **[diagnostics.md](archive/diagnostics.md)**. Pre-pack diagnostics on motor / tectum / hippocampal failures.
- **[whats_in_the_box.md](archive/whats_in_the_box.md)**. Earlier plain-language summary, superseded by `03_results.md`.

## See also

- **[Top-level README](../README.md)**. Install + quickstart
- **[notebooks/](../notebooks/)**. Interactive walkthroughs (load π, click regions, see top-K)
- **[pipeline/README.md](../pipeline/README.md)**. Reproduction script map
- **[experiments/README.md](../experiments/README.md)**. Per-pack and ablation runners
- **[outputs/README.md](../outputs/README.md)**. Generated artefact catalogue
