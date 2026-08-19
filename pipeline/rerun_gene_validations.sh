#!/usr/bin/env bash
# Rebuild mouse_genes.npy and re-run every validation that consumes it.
#
# Allen MetaImage .raw buffers are column-major and must be reshaped with
# order="F". The read is done in:
#   - pipeline/00_external/02_mouse_genes.py
#   - experiments/autism_subtypes/allen_expansion/download_pagani_ish.py
#
# The production π is gene-free (use_gene_gw=False), so π and the FC/SC
# validations do not depend on the gene matrix and are not re-run here.
#
# Prereqs: the otter env (numpy/scipy/pandas/anndata/nibabel/allensdk), the v2
# mouse .mat present, and the Allen ISH energy zips cached under
# experiments/autism_subtypes/allen_expansion/pagani_ish_cache/.
#
# Usage:  bash pipeline/rerun_gene_validations.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

echo "==> [1/4] Rebuild mouse_genes.npy from the ISH volumes"
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
  * Check the new mouse_genes.npy: Th should peak in olfactory bulb +
    midbrain, Mbp in white-matter tracts.
  * Update the gene-validation docs from the regenerated logs.
------------------------------------------------------------------------------
NOTE
