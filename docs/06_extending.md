# Extending OTTER

Extensions should preserve the distinction between within-species relational
costs, cross-species costs and anatomical supervision.

## Add a relational modality

A new within-species modality must provide one finite, symmetric,
zero-diagonal distance matrix per species. Add the cost function under
src/otter/costs/, store the two matrices in the cost bundle and pass them to a
model class that records their weights in its configuration. Add synthetic
tests for symmetry, scale, missing-data handling and shape.

Do not alter the canonical recipe silently. A new modality is an experimental
model until its fitting rule, hyperparameters and evaluation are documented and
all dependent result logs have been regenerated.

## Add a regional correspondence entry

Create a module under src/otter/data/anchor_packs/ exposing a builder that
returns RegionAnchorEntry objects. Each entry needs:

- non-conflicting pair IDs;
- explicit mouse and human parcel sets;
- a primary comparative-anatomy source;
- tests that verify both parcel sets and bilateral handling where applicable.

Add the module to src/otter/data/anchor_packs/registry.py only if it is intended
to change the canonical fit. Registry changes require refitting the coupling and
rerunning every dependent analysis.

## Add a species

The new species needs parcel metadata, group-level functional connectivity and
the within-species costs used by the chosen model. It also needs a defensible
cross-species anatomical frame; the mouse–human Garin anchors cannot be reused
for a different species pair.

The model API accepts AnnData-like objects with parcel metadata in .var and
connectivity in .uns. Species-specific loading and coordinate transforms belong
in the data layer rather than in the solver.

## Add an evaluation

Put substantive analyses in a named subdirectory of experiments/. Each
producer should:

- load the coupling with load_pi() or record the complete refit recipe;
- stamp the output with the coupling hash or refit provenance;
- keep target masks and normalisation rules explicit;
- recompute the tested statistic inside every null permutation;
- write machine-readable output beneath outputs/logs/.

If the evaluation reuses a fitting input, anatomical scaffold or
hyperparameter-selection target, describe it as an internal consistency or
sensitivity analysis rather than independent validation.
