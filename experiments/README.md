# OTTER analyses

This directory contains analysis and sensitivity scripts. The
reusable fitting, data and evaluation APIs live under `src/otter/`; scripts here read the released
coupling or refit a clearly specified model configuration.

## Analysis directories

| Directory | Scope |
|---|---|
| `section1_stability/` | Coupling stability |
| `section2_supervision/` | Cost-term decomposition and supervision-withheld recovery |
| `margulies_2016_principal_gradient/` | Principal functional-gradient transfer |
| `fulcher_2019_multimodal_gradient/` | Microstructural and cytoarchitectural transfer |
| `biccn_2023_cell_types/` | Cell-class marker-expression transfer |
| `hodge_2019_cortical_layers/` | Laminar marker-expression transfer |
| `coletta_2020_cross_species_rsn/` | Functional-network correspondence |
| `transbrain_2025_benchmark/` | Comparative-method analysis |
| `section5_coverage_rigor/` | Mouse-based reconstruction of human connectivity |
| `reverse_translation/` | Human-to-mouse translation, clinical disease dimensions and target ranking |
| `validation/` | Checks against published map resources |

`anchor_packs/` contains the regional-entry definitions and source-specific runners;
`ablations/` contains additional model sensitivities. Other named directories contain supporting
or application-specific analyses and can be run independently.

The canonical coupling uses 21 Garin homology classes and 26 curated regional entries. The
19 scorable Beauchamp region pairs provide a common scoring frame and inform hyperparameter
evaluation. They are not supplied as anatomical correspondence constraints, but some benchmark
territories overlap the anatomical scaffold, so they should not be described as wholly independent
validation.

Run scripts from the repository root with `PYTHONPATH=src`. Outputs are written beneath
`outputs/logs/`, `outputs/coupling/` or `outputs/figures/` as documented by each script. Analysis
logs should record the input coupling filename and SHA-256; verify these against `pi_provenance()`
before comparing results.
