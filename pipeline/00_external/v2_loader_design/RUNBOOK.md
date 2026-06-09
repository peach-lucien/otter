# v2 End-to-End Runbook

> Step-by-step commands to refit HOMER under the v2 corrs_mouse table.
> Designed to be runnable on your local machine where `ot` (Python
> Optimal Transport), AllenSDK, Jupyter, and the full validation
> dependencies are available. Sandbox couldn't run the heavy steps.
>
> All paths are relative to the project root:
> `/Users/Peach_R/Dropbox/Work/ResearchProjects/brain_crossspecies_translation/homer/`

## 0. Sanity check before starting

```bash
cd /Users/Peach_R/Dropbox/Work/ResearchProjects/brain_crossspecies_translation/homer

# Confirm v2 file is in place
ls -la ../data_crossspecies/updated_connectom_0906_26/corrs_mouse_v2.mat

# Confirm the v2 loader test suite passes
HOMER_TEST_FAST=1 pytest tests/test_v2_loader.py -v
# Expect: 22 passed, 1 skipped

# Confirm the existing test suite still passes under v2
pytest tests/ -x --no-header
# Expect: existing tests still pass; one new test file added (test_v2_loader.py)
```

## 1. Verify anchor packs are stable v1 → v2

Anchor packs that use small-radius xyz lookups may shift slightly under
v2 (the 0.117 mm centre drift between v1 and v2 SS frames can cross
voxel boundaries). Run the diff diagnostic to enumerate any membership
changes:

```bash
PYTHONPATH=src python pipeline/00_external/diff_anchor_packs_v1_vs_v2.py
# Output: data_external/_diagnostics/anchor_pack_v1_v2_diff.json
# Console: per-pack Jaccard similarity; entries with Jaccard < 0.95
# get flagged for manual review.
```

**Expected**: most packs should have Jaccard ≥ 0.99 (≤ 1 parcel shifted).
Small-radius packs (PAG, LC, raphe, sub-regions of cortex) may have
Jaccard in the 0.90–0.99 range — review those individually before
shipping. Anything below 0.90 needs investigation.

If a pack changes materially, decide:
- Accept v2 membership (recommended — Paul's warp is more accurate)
- Pin to v1 set explicitly in the pack module
- Adjust pack radius to recover the v1 set

## 2. Rebuild gene matrix under v2

```bash
PYTHONPATH=src python pipeline/00_external/02c_mouse_genes_v2.py
```

This reads `ns_voxel_indices` from the v2 t-table (already pre-warped
by Paul into CCFv3 25 µm space), downsamples to 200 µm, and samples
all 61 Allen ISH genes from the Pagani cache (or downloads missing).

**Outputs**:
- `data_external/mouse_genes.npy` — shape (1864, 61)
- `data_external/mouse_gene_list.csv`
- `data_external/mouse_genes_meta.json`

**Expected runtime**: ~2 minutes if cache is warm; up to 5 minutes if
several genes need re-download.

The legacy `02_mouse_genes.py` / `02b_mouse_genes_direct.py` paths are
preserved for v1 compatibility but will fail loudly if pointed at v2
data (B5 fix in `_mouse_transform.py`).

## 3. Rebuild SC matrix under v2

```bash
PYTHONPATH=src python pipeline/00_external/01b_mouse_sc_v2.py
```

This reads `ns_center_ix` (parcel centre voxel in CCFv3 25 µm),
downsamples to 100 µm to query AllenSDK's annotation, maps fine label
→ summary structure via the structure tree, then indexes the
summary-level SC matrix.

**Outputs**:
- `data_external/mouse_sc.npy` — shape (1864, 1864) float32
- `data_external/mouse_sc_meta.json`

**Expected runtime**: ~10–30 minutes (AllenSDK fetches all Oh 2014
unionised projection volumes; the structure-level SC build is the
heavy step). If AllenSDK cache is already populated, much faster.

## 4. Build the v2 mouse AnnData (cached)

**Run this BEFORE step 5** — `03_build_costs.py` calls `load_cached()` which
requires a schema-tagged cache. Run the rebuild here:

```bash
# Remove any pre-existing caches (if you have v1 caches lying around the
# load_cached strict-schema check would refuse them):
rm -f outputs/anndata/mouse.h5ad outputs/anndata/mouse.fc.npy outputs/anndata/mouse.voxels.npz

PYTHONPATH=src python -c "
from homer.data import build_anndata
build_anndata('mouse',
              cache_dir='outputs/anndata',
              overwrite=True,
              cache_voxels=True)
"
```

For the human cache, the loader now treats legacy (untagged) v1 caches as
acceptable — no rebuild needed unless your h5ad is genuinely stale. If
you do need to rebuild:

```bash
rm -f outputs/anndata/human.h5ad outputs/anndata/human.fc.npy outputs/anndata/human.voxels.npz
PYTHONPATH=src python -c "
from homer.data import build_anndata
build_anndata('human',
              cache_dir='outputs/anndata',
              overwrite=True,
              cache_voxels=True)
"
```

## 5. Rebuild aligned gene + ortholog matrix

```bash
PYTHONPATH=src python pipeline/03_build_costs.py
```

Re-runs the gene-ortholog alignment between mouse and human gene
matrices. Reads the freshly-written `mouse_genes.npy` and `human_genes.npy`
**and the freshly-rebuilt AnnDatas from step 4**, writes:
- `data_external/mouse_genes_aligned.npy`
- `data_external/human_genes_aligned.npy`

## (was step 5, now folded into step 4)

```bash
PYTHONPATH=src python -c "
from homer.data import build_anndata
build_anndata('mouse',
              cache_dir='outputs/anndata',
              overwrite=True,
              cache_voxels=True)
"
```

This is the load-bearing artefact. The cached `outputs/anndata/mouse.h5ad`
now carries `A.uns['mouse_schema'] = 'v2'`, the new region label
columns in `A.var`, and the NS / SS voxel index lists in `A.uns`.

**Important**: any pre-existing `outputs/anndata/mouse.h5ad` from v1
must be deleted first, OR `load_cached("mouse")` will raise
`CacheSchemaMismatch` and tell you exactly which `rm` command to run.

Same for human, no change needed:

```bash
PYTHONPATH=src python -c "
from homer.data import build_anndata
build_anndata('human',
              cache_dir='outputs/anndata',
              overwrite=True,
              cache_voxels=True)
"
```

## 6. π refit cascade

```bash
PYTHONPATH=src python pipeline/run_recommended_model.py
```

This orchestrator solves → composes → bootstraps → trust → GUI.
Needs `ot` (POT package). Reads the freshly-built AnnData, gene
matrices, and SC matrices. Writes:
- `outputs/coupling/pi_fc_plus_SC_with_all_packs.npy`
- `outputs/trust/trust_multisource_all_packs.npz`
- `docs/index.html` (rebuilt GUI)

**Expected runtime**: 30–60 minutes depending on iteration budget.

## 7. Re-run the validation suite

```bash
# One-shot all 13 validation scripts. Each writes its own JSON output
# under outputs/logs/. The notebooks read those JSONs.

# Beauchamp (cross-species pairs)
PYTHONPATH=src python pipeline/05f_beauchamp_validation.py

# Cell-type, layer, gene-spatial validations
PYTHONPATH=src python experiments/biccn_2023_cell_types/01_cell_type_validation.py
PYTHONPATH=src python experiments/hodge_2019_cortical_layers/01_layer_marker_validation.py
PYTHONPATH=src python experiments/hodge_2019_cortical_layers/02_layer_marker_refined.py

# Gradient and network validations
PYTHONPATH=src python experiments/margulies_2016_principal_gradient/01_gradient_validation.py
PYTHONPATH=src python experiments/fulcher_2019_multimodal_gradient/01_gradient_validation.py
PYTHONPATH=src python experiments/coletta_2020_cross_species_rsn/01_correspondence_validation.py
PYTHONPATH=src python experiments/enigma_cross_disorder/01_per_disorder_prediction.py

# Autism subtypes (the gene-spatial set)
PYTHONPATH=src python experiments/autism_subtypes/04_subtype_translation.py
PYTHONPATH=src python experiments/autism_subtypes/09_gene_spatial_translation.py

# TransBrain head-to-head benchmark
PYTHONPATH=src python experiments/transbrain_2025_benchmark/01_transbrain_benchmark.py

# Falsification tests (negative controls)
PYTHONPATH=src python experiments/balsters_2020_mfc_divergence/01_mfc_divergence.py
PYTHONPATH=src python experiments/buckner_krienen_2013_tethering/01_tethering_test.py
# (TransBrain benchmark requires `pip install transbrain` first)
```

**Expected**: each runs in a few minutes. The Beauchamp top-1 and the
TransBrain centroid distance are the headline numbers; track those
before vs after.

The ABIDE test (`experiments/autism_subtypes/abide_subtype_prediction.py`)
needs a 3–8 GB nilearn download — only run if you want fresh ABIDE numbers.

## 8. Re-execute notebooks

```bash
# Re-execute notebooks 01–15 to pick up the refitted π and new validation JSONs.
# nbconvert in-place execution; suppresses cell output regeneration.
for nb in notebooks/*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

**Expected runtime**: ~30 minutes for all 15 notebooks.

After execution, manually audit the markdown cells in each notebook —
notebook execution refreshes code outputs but NOT markdown prose. Any
notebook whose interpretive cells quote specific numbers (e.g.,
"Margulies |r| = 0.426") needs the prose updated to match the new
output. This is the manual cleanup step that always lags an automated
re-execution.

## 9. Sweep docs for stale numbers

```bash
# Search docs/ for any numeric value that's likely to have shifted:
grep -rEn "(r|ρ|p)\s*=\s*[+-]?[0-9.]+|top[-_]?[15]" docs/ experiments/*/README.md README.md \
  | sort -u
```

Anything that points to a specific quantity (correlation r, p-value,
top-k accuracy, parcel count, percentage agreement) should be checked
against the refreshed JSONs. The previous v1 docs cleanup pass took
two days of manual sweeping — be patient.

## 10. Cleanup (optional, do last)

After you're satisfied with the v2 cascade:

```bash
# Retire the warp_rebuild outputs (superseded by v2 t-table):
git mv data_external/_warp_rebuild data_external/_archive_warp_rebuild_pre_v2
git mv data_external/mouse_genes_warped.npy data_external/_archive/

# Retire the heuristic-transform diagnostic:
git mv data_external/_diagnostics/mouse_to_ccf_transform.json \
       data_external/_diagnostics/_archive_mouse_to_ccf_transform_v1.json

# (Optional) move the old 02_mouse_genes / 01_mouse_sc into a legacy folder
mkdir -p pipeline/00_external/_v1_legacy
git mv pipeline/00_external/00c_align_mouse_to_ccf.py    pipeline/00_external/_v1_legacy/
git mv pipeline/00_external/_mouse_transform.py          pipeline/00_external/_v1_legacy/
git mv pipeline/00_external/02_mouse_genes.py            pipeline/00_external/_v1_legacy/
git mv pipeline/00_external/02b_mouse_genes_direct.py    pipeline/00_external/_v1_legacy/
git mv pipeline/00_external/01_mouse_sc.py               pipeline/00_external/_v1_legacy/
```

Don't move these until the v2 cascade is fully working and validated —
they're useful as a reference and for the human-side v1 path.

## 11. Final commit

```bash
git add -A
git commit -m "v2 cascade complete: corrs_mouse_v2 + refitted π + re-validated"
```

---

## Headline numbers to track before/after

| Metric | v1 number (pre-refit) | v2 expected (after) |
|---|---|---|
| Beauchamp top-1 (anchor-overlapping) | 40.3% | unchanged or slight increase |
| TransBrain centroid distance | 23.6 mm | similar |
| Margulies principal gradient \|r\| | 0.408 | similar |
| Pagani r | +0.548 | similar |
| Fulcher gradient r | +0.230 / +0.305 | similar |
| ENIGMA r | +0.988 | similar |
| Hodge upper-layer r | +0.92 | similar |
| Balsters falsification (% mass) | 0% | should remain 0% |

Headline shifts of more than 0.05 in correlations or more than 5 % in
top-K should be investigated — they probably indicate a parcel-set
shift (B3 anchor-pack drift) that's worth understanding rather than
just accepting.

## What's NOT in this runbook

- AnnData migration for already-cached h5ad files. The `load_cached`
  schema check will refuse to load a v1 cache when v2 is on disk; you
  must rebuild with `build_anndata(overwrite=True)`.
- Notebook markdown audit (step 8 manual sweep). This always lags
  automation.
- ABIDE refresh (3–8 GB download). Skip unless you need fresh ABIDE
  numbers in the paper.
- Docs restructure (the previous "part 1" item from the package cleanup
  handoff). Separate task; not blocked by v2.

## Troubleshooting

- **`CacheSchemaMismatch` raised on load**: delete the listed files,
  rebuild AnnData with `overwrite=True`. The error message includes
  the exact `rm` command.
- **`colleague_voxel_to_ccf_world` raises ValueError**: caller is using
  a v1-only function with v2 data. Migrate to `ns_voxel_indices` +
  NS affine.
- **`unrecognised ht schema`**: the .mat file you pointed at doesn't
  match either v1 or v2 exactly. Most likely a future variant; update
  `_V2_HT` in `src/homer/data/io.py` first (and pin a new schema
  detection per the spec).
- **Anchor pack Jaccard < 0.9**: small-radius pack shifted under v2.
  Decide whether to accept v2 membership (recommended) or pin to v1.
