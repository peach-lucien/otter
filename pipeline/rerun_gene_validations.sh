#!/usr/bin/env bash
# Re-run the gene-spatial pipeline + validations after the ISH volume-read fix.
#
# WHY: `read_ish_grid` was reshaping the Allen MetaImage .raw buffer in C-order;
# MetaImage is column-major, so every mouse ISH gene volume was read spatially
# SCRAMBLED. Fixed to order="F" in:
#   - pipeline/00_external/02_mouse_genes.py
#   - experiments/autism_subtypes/allen_expansion/download_pagani_ish.py
#
# This rebuilds mouse_genes.npy with the corrected read and re-runs every
# validation that consumes it. The production π is gene-free (use_gene_gw=False),
# so π and all FC/SC validations are UNAFFECTED and are not re-run here.
#
# Prereqs: the homer env (numpy/scipy/pandas/anndata/nibabel/allensdk), the v2
# mouse .mat present, and the Allen ISH energy zips cached under
# experiments/autism_subtypes/allen_expansion/pagani_ish_cache/ (already present).
#
# Usage:  bash pipeline/rerun_gene_validations.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

echo "==> [1/4] Rebuild mouse_genes.npy with the F-order ISH read"
python pipeline/00_external/02_mouse_genes.py

echo "==> [2/4] Re-align orthologs (re-saves *_aligned.npy from the fixed genes)"
python pipeline/00_external/05_orthologs.py

echo "==> [3/4] Re-run the gene-spatial validations"
python experiments/biccn_2023_cell_types/01_cell_type_validation.py
python experiments/hodge_2019_cortical_layers/01_layer_marker_validation.py
python experiments/hodge_2019_cortical_layers/02_layer_marker_refined.py
python experiments/autism_subtypes/09_gene_spatial_translation.py

echo "==> [4/4] Done. Updated logs:"
ls -la outputs/logs/biccn_2023_cell_types.json \
        outputs/logs/hodge_2019_layer_markers.json \
        outputs/logs/autism_subtypes_gene_spatial.json 2>/dev/null || true

cat <<'NOTE'

------------------------------------------------------------------------------
Next:
  * Sanity-check the new mouse_genes.npy: Th should peak in olfactory bulb +
    midbrain, Mbp in white-matter tracts (the bug made every gene centre-blurred).
  * The numbers in these logs supersede the current docs/notebooks. The
    gene-validation docs are intentionally left in a "pending re-run" holding
    state until these complete; update them from the refreshed logs.

OPTIONAL (separate methodology decision. NOT applied here):
  The gene validations route mouse->human with the bare un-normalised sum
  `score @ pi`, whereas Margulies/TransBrain use the coverage-normalised
  transport-weighted average. To make routing consistent, divide by the column
  mass in each script after loading pi:

      colmass = np.maximum(pi.sum(axis=0), 1e-12)
      pred    = (score @ pi)        / colmass          # observed
      pred_n  = (score @ pi[perm])  / colmass          # permuted-pi null
                                                       # (row-perm leaves colmass unchanged)

  Lines: BICCN 01 (146/154), Hodge 01 (129/143), Hodge 02 (116/130/179),
  autism 09 (111/112). Run bug-fix-only FIRST to isolate the bug's effect,
  then apply this and re-run if you want the consistent methodology.
------------------------------------------------------------------------------
NOTE
