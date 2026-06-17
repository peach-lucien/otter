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

VERSION="v1.0.0"
# Stop macOS bsdtar from writing AppleDouble (._*) sidecars into the archives.
export COPYFILE_DISABLE=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
OUT_DIR="$(dirname "$ROOT")"

# ---- Archive 1: reproduce bundle -------------------------------------------
REPRODUCE=(
  # HOMER-generated (we own these)
  outputs/coupling/pi_fc_plus_SC_with_all_packs.npy
  outputs/coupling/trust_multisource_all_packs.npz
  outputs/coupling/per_disorder_predictions.npz
  outputs/anndata/mouse.h5ad
  outputs/anndata/human.h5ad
  outputs/anndata/mouse.voxels.npz
  outputs/anndata/mouse_voxel_counts.npy
  outputs/anndata/human_voxel_counts.npy
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

build "homer-reproduce-${VERSION}.tar.gz" "${REPRODUCE[@]}"

# ---- Archive 2: full raw inputs (optional, for a from-scratch rebuild) ------
# The whole data_external/ except the regeneratable Allen ISH download caches.
echo ">> building homer-raw-inputs-${VERSION}.tar.gz (full data_external/)"
tar --no-xattrs --exclude='.DS_Store' --exclude='data_external/_ish_cache' \
    --exclude='data_external/_ish_cache_v2' \
    -czf "$OUT_DIR/homer-raw-inputs-${VERSION}.tar.gz" data_external
echo "   wrote $OUT_DIR/homer-raw-inputs-${VERSION}.tar.gz  ($(du -h "$OUT_DIR/homer-raw-inputs-${VERSION}.tar.gz" | cut -f1))"
shasum -a 256 "$OUT_DIR/homer-raw-inputs-${VERSION}.tar.gz"

echo
echo "Done. Upload both .tar.gz files to one Zenodo record, then paste the DOI,"
echo "URLs and checksums into data_manifest.json (see DATA.md)."
