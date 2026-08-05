#!/usr/bin/env bash
# Build the two Zenodo archives from the repo root.
# Run this on the machine that has the real data_external/ and outputs/.
#
#   bash scripts/build_archives.sh
#
# Produces (next to the repo):
#   ../homer-reproduce-v1.0.0.tar.gz   (~173 MB gzipped)  -> Zenodo Archive 1
#   ../homer-raw-inputs-v1.0.0.tar.gz  (~606 MB gzipped)  -> Zenodo Archive 2
# and prints the sha256 of each so you can record it.
#
# Paths inside the tarballs are repo-relative, so scripts/fetch_data.py unpacks
# them cleanly at the repo root. Missing entries are skipped with a warning.
set -uo pipefail

REPRODUCE_VERSION="v1.3.0"   # ships every data file the notebooks/experiments load (full audit).
# v1.3.0 adds pi_canonical.npy, pi_canonical_sharp.npy and
# trust_multisource_canonical.npz, which v1.2.0 omitted.
RAW_VERSION="v1.0.0"         # unchanged content
# Stop macOS bsdtar from writing AppleDouble (._*) sidecars into the archives.
export COPYFILE_DISABLE=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
OUT_DIR="$(dirname "$ROOT")"

# ---- Archive 1: reproduce bundle -------------------------------------------
REPRODUCE=(
  # --- canonical: what load_pi() returns and what every notebook needs. Added 2026-07-20;
  # the bundle previously shipped only the retired pre-warp couplings, so a fresh user who ran
  # fetch_data.py and opened any notebook hit FileNotFoundError on the first cell.
  outputs/coupling/pi_canonical.npy
  outputs/coupling/pi_canonical_sharp.npy
  outputs/coupling/trust_multisource_canonical.npz
  # OTTER-generated (we own these), all coupling files the notebooks +
  # experiments load (recommended pi, strict pi, ablation variants, trust maps).
  outputs/coupling/pi_fc_plus_SC_with_all_packs.npy
  outputs/coupling/pi_fc_plus_SC.npy
  outputs/coupling/pi_fc_plus_SC_xyz_zero.npy
  outputs/coupling/pi_fc_plus_SC_with_M1.npy
  outputs/coupling/pi_fc_plus_SC_with_amygdala.npy
  outputs/coupling/pi_fc_plus_SC_with_biccn_motor.npy
  outputs/coupling/pi_fc_plus_SC_with_cingulate.npy
  outputs/coupling/pi_fc_plus_SC_with_cingulate_rsc_only.npy
  outputs/coupling/pi_fc_plus_SC_with_entorhinal.npy
  outputs/coupling/pi_fc_plus_SC_with_hippocampal.npy
  outputs/coupling/pi_fc_plus_SC_with_hippocampal_subi_only.npy
  outputs/coupling/pi_fc_plus_SC_with_lateral_pfc.npy
  outputs/coupling/pi_fc_plus_SC_with_olfactory.npy
  outputs/coupling/pi_fc_plus_SC_with_olfactory_pir_only.npy
  outputs/coupling/pi_fc_plus_SC_with_striatum.npy
  outputs/coupling/pi_fc_plus_SC_with_tectum.npy
  outputs/coupling/pi_fc_plus_SC_with_tectum_sc_only.npy
  outputs/coupling/pi_fc_plus_SC_with_pag.npy
  outputs/coupling/pi_fc_plus_SC_with_visual.npy
  outputs/coupling/pi_fc_plus_SC_per_region_xyz_v2.npy
  outputs/coupling/trust_multisource_all_packs.npz
  outputs/coupling/trust_score_fc_plus_SC.npz
  outputs/coupling/trust_score_fc_plus_SC_with_M1_hippo.npz
  outputs/coupling/bootstrap_aggregate_fc_plus_SC.npz
  outputs/coupling/per_disorder_predictions.npz
  outputs/anndata/mouse.h5ad
  outputs/anndata/human.h5ad
  outputs/anndata/mouse.voxels.npz
  outputs/anndata/mouse_voxel_counts.npy
  outputs/anndata/human_voxel_counts.npy
  outputs/anndata/full_costs.npz
  outputs/anndata/_schaefer_order.txt
  # volumetric references for the glass-brain panels (Fig 1a-c/g, Fig 3b). outputs/coupling/ is
  # gitignored, so these are only available through the archive; without them the volumetric
  # panel scripts fail even though every statistic still computes.
  outputs/coupling/mouse_parcel_filled_100um.nii.gz
  outputs/coupling/mouse_parcel_labels_25um.nii.gz
  outputs/coupling/mouse_tpl_100um.nii.gz
  # third-party-derived validation inputs (confirm redistribution rights)
  data_external/human_genes.npy
  data_external/human_gene_list.csv
  data_external/mouse_genes.npy
  data_external/mouse_gene_list.csv
  data_external/human_sc.npy
  data_external/human_sc_meta.json
  data_external/mouse_sc.npy
  data_external/mouse_sc_meta.json
  data_external/mouse_sc_knox_augmented.npy
  data_external/knox_sc
  data_external/orthologs.csv
  data_external/orthologs_meta.json
  data_external/fulcher_2019_gradients
  data_external/pagani_2026
  data_external/transbrain_2025
  data_external/_domhof_extracted
  data_external/_diagnostics
  data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc
  data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/imaging/DSURQE_40micron_R_mapping_long.csv
  data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/DSURQE_tree.json
  data_external/p6ebec-hbp-d000038_SC-FC_HCP_eNKI_pub/Schaefer2018_400Parcels_17Networks.zip
)

build () {
  local name="$1"; shift
  local listfile; listfile="$(mktemp)"
  local missing=0
  for p in "$@"; do
    if [ -e "$p" ]; then echo "$p" >> "$listfile"; else echo "  WARN missing: $p" >&2; missing=$((missing+1)); fi
  done
  echo ">> building $name  ($(wc -l < "$listfile") entries, $missing missing)"
  tar --no-xattrs --exclude='.DS_Store' -czf "$OUT_DIR/$name" -T "$listfile"
  rm -f "$listfile"
  echo "   wrote $OUT_DIR/$name  ($(du -h "$OUT_DIR/$name" | cut -f1))"
  shasum -a 256 "$OUT_DIR/$name"
}

build "homer-reproduce-${REPRODUCE_VERSION}.tar.gz" "${REPRODUCE[@]}"

# ---- Archive 2: full raw inputs (optional; content unchanged from v1.0.0) ----
# Only rebuilt when BUILD_RAW=1, since the raw inputs haven't changed and the
# existing v1.0.0 file on Zenodo is still valid.
if [ "${BUILD_RAW:-0}" = "1" ]; then
  echo ">> building homer-raw-inputs-${RAW_VERSION}.tar.gz (full data_external/)"
  tar --no-xattrs --exclude='.DS_Store' --exclude='data_external/_ish_cache' \
      --exclude='data_external/_ish_cache_v2' \
      -czf "$OUT_DIR/homer-raw-inputs-${RAW_VERSION}.tar.gz" data_external
  echo "   wrote $OUT_DIR/homer-raw-inputs-${RAW_VERSION}.tar.gz  ($(du -h "$OUT_DIR/homer-raw-inputs-${RAW_VERSION}.tar.gz" | cut -f1))"
  shasum -a 256 "$OUT_DIR/homer-raw-inputs-${RAW_VERSION}.tar.gz"
else
  echo ">> skipping raw-inputs archive (unchanged; set BUILD_RAW=1 to rebuild it)"
fi

echo
echo "Done. Upload homer-reproduce-${REPRODUCE_VERSION}.tar.gz as a NEW VERSION of the"
echo "Zenodo record, then give the new record id so the manifest can be refreshed."
