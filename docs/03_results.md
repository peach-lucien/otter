# Analysis guide

The manuscript is the authoritative narrative for the reported results. This
page identifies the current analysis groups and the interpretation boundary for
each one without duplicating result tables.

## 1. Coupling organisation

The coupling is assessed for concentration, broad areal organisation and
topographic preservation. Coupling concentration depends on the solver path and
stopping point, so a sharp row is not a calibrated measure of confidence.

Relevant code and logs:

- experiments/coupling_summary/
- outputs/logs/coupling_summary_canonical.json
- outputs/logs/fig1_coupling_matrix.json

## 2. Cost terms and anatomical supervision

The Beauchamp transcriptomic correspondences provide a common 19-region scoring
frame. They are not explicit OTTER constraints, but their anatomy overlaps parts
of the anchor-warped scaffold and several regional entries. Model decomposition
is therefore descriptive. Generalisation is assessed by refitting after removing
the supervision that overlaps each target, and by leaving out each Garin class or
regional entry in turn.

Relevant code and logs:

- experiments/section2_supervision/
- outputs/logs/heldout_three_config_canonical.json
- outputs/logs/out_a1b_loro.json

## 3. Cross-species transfer

Mouse maps are routed through the coupling and compared with human maps. The
anchor-derived networks and connectome-derived principal gradient are internal
consistency tests because they reuse supervision or connectomes entering OTTER.
Microstructural and marker-expression analyses test cross-modal transfer, but
several maps follow a shared areal hierarchy and should not be counted as fully
independent validations. Cell-class maps are marker-expression proxies, not
single-cell abundance estimates.

Relevant directories:

- experiments/coletta_2020_cross_species_rsn/
- experiments/margulies_2016_principal_gradient/
- experiments/fulcher_2019_multimodal_gradient/
- experiments/biccn_2023_cell_types/
- experiments/hodge_2019_cortical_layers/

## 4. Comparative method analysis

OTTER and TransBrain are compared on their shared Brainnetome region-level
scoring ground. Differences in resolution, inputs and outputs are described as
method capabilities rather than as a single superiority score.

Relevant code and logs:

- experiments/transbrain_2025_benchmark/
- outputs/logs/transbrain_benchmark_summary.json

## 5. Mouse-based reconstruction of human connectivity

Human functional connectivity is reconstructed as
pi_col.T @ mouse_fc @ pi_col, where columns of pi are normalised before the
push-forward. This is a connectional reconstruction measure, not anatomical
assignment and not the raw column mass of the semirelaxed coupling.

Relevant code and logs:

- experiments/section5_coverage_rigor/
- outputs/logs/section5_reconstruction_coverage.json
- outputs/logs/fig5_panel_values.json

## 6. Translation utilities

Forward routing turns a mouse map into a human spatial hypothesis; reverse
routing ranks mouse structures for a human target. The demonstrations establish
the mechanics and show examples of cross-modal consistency. They do not by
themselves establish disease validity or experimental efficacy.

Relevant code and logs:

- experiments/reverse_translation/
- experiments/autism_subtypes/
- outputs/logs/pagani_subtype_translation_corrected.json

## Provenance

Use otter.data.load_pi() to load the released pi_canonical.npy and
otter.data.pi_provenance() to obtain its SHA-256. Result logs identify either the
coupling file and hash that they loaded or the recipe of a coupling refitted
inside the analysis. Do not combine a result log with a different coupling.
