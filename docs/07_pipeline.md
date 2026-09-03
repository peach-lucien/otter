# Reproduction workflow

Run commands from the repository root in the otter environment.

## Use the released coupling

Most users should download the release bundle rather than refit the model:

    python scripts/fetch_data.py

This provides outputs/coupling/pi_canonical.npy, the processed parcel tables and
the inputs used by the notebooks and reproducible analyses. load_pi() loads the
released coupling by default.

## Rebuild the inputs

The raw tier is required only for a from-scratch rebuild:

    python scripts/fetch_data.py --tier raw
    PYTHONPATH=src python pipeline/00_external/01_mouse_sc.py
    PYTHONPATH=src python pipeline/00_external/02_mouse_genes.py
    PYTHONPATH=src python pipeline/00_external/03_human_sc.py
    PYTHONPATH=src python pipeline/00_external/04_human_genes.py
    PYTHONPATH=src python pipeline/00_external/05_orthologs.py
    PYTHONPATH=src python pipeline/02_build_anndata.py
    PYTHONPATH=src python pipeline/03_build_costs.py

Source-specific requirements are documented in
[pipeline/00_external/README.md](../pipeline/00_external/README.md).

## Refit the canonical coupling

    PYTHONPATH=src python pipeline/run_recommended_model.py \
      --output outputs/coupling/pi_canonical_refit.npy

The command uses the recipe in otter.repro: functional and structural
connectivity, the anchor-warped spatial cost, 21 Garin homology classes and 26
curated regional entries. It writes a provenance sidecar next to the refitted
coupling and does not overwrite the released file unless explicitly requested.

## Reproduce analyses

Analyses are grouped by purpose under experiments/; see
[03_results.md](03_results.md) for the directory and log associated with each
section. Some scripts refit the model for holdouts or nulls, while others load
the released coupling. Their result logs record the relevant coupling hash or
refit recipe.

The notebooks provide the shortest executable walkthrough:

    jupyter lab notebooks/

## Build the explorer

    PYTHONPATH=src python pipeline/08_build_gui.py --publish

This writes outputs/gui/index.html and copies the self-contained explorer to
docs/index.html. The explorer's categories are display metadata rather than
confidence or validation tiers.
