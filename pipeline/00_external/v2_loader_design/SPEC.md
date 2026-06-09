# Implementation Spec: `corrs_mouse_v2.mat` Support in `homer.data.io`

> Authored by the Plan agent (2026-06-09). Adversarially reviewed in
> `REVIEW.md` alongside this file. Read both before implementing.

## Scope and constraint

This spec covers the modifications needed to `src/homer/data/io.py` (and a few satellite places) to support the new v2 mouse file (`corrs_mouse_v2.mat`) without breaking the v1 path for either `corrs_mouse.mat` or `corrs_human.mat`.

**Human is not affected.** `corrs_human.mat` retains the v1 7-column `ht` and v1 semantics. Only the mouse path branches. The spec must keep `load_struct`, `load_metadata`, and `parse_t_table` species-symmetric in their public signatures.

---

## 1. Schema detection

### Recommendation: detect v2 from the `ht` column list, not from `m.version`

Rationale:

- `m.version` exists in v2 but is a free-form string ("v2", "2.0", whatever Paul writes). Matching on it is brittle.
- The `ht` cell array is the schema. Every consumer of `t` cross-references `ht` to find columns. Discriminating on `ht` is operationally equivalent to "what columns exist."
- `m.version` *is* still useful for logging and as a tie-breaker, but should not be the primary signal.

### Decision rule (in `parse_t_table`, applied to the `ht` argument):

```
v1_HT = ["type", "numid", "pairid", "region", "subregion", "center", "indices"]
V2_HT = [
    "type", "numid", "pairid", "region", "subregion",
    "AS_center_mm", "AS_ix", "AS_center_ix",
    "AS_region_center_DSURQUE", "AS_region_center_ABA",
    "AS_region_vote_DSURQUE",   "AS_region_vote_ABA",
    "DS_center_mm", "DS_ix", "DS_center_ix",
    "DS_region_center_DSURQUE", "DS_region_center_ABA",
    "DS_region_vote_DSURQUE",   "DS_region_vote_ABA",
]
```

Required: `list(ht) == V1_HT` exact match → v1 path; `list(ht) == V2_HT` exact match → v2 path; anything else → `ValueError("unrecognised ht schema: {ht}")`.

**Do not** accept a "v2-ish" file with permuted columns or extra columns.

### Why not "presence of `AS_center_mm`"?

Substring/membership checks (e.g., `"AS_center_mm" in ht`) would silently pass on a partial migration. Full-list equality forces every column to be exactly what we expect.

### Where the decision is made

- `parse_t_table(t, ht)` is the schema-decision point.
- Add a private constant `_V1_HT`, `_V2_HT` in `io.py`, and a private helper `_detect_schema(ht: list[str]) -> Literal["v1", "v2"]`.

**Cross-check assertion (assumption to be verified):** Paul's `corrs_human.mat` `ht` is still the v1 7-column list. Verify by reading `corrs_human.mat` head-of-file before relying on this.

---

## 2. `_mat_path` change

### Current behaviour

```python
p = data_dir / f"corrs_{species}.mat"
```

### Recommendation

Promote `_mat_path` to a resolver that returns both the path and the detected schema version.

#### Resolution rule (for `species == "mouse"`):

1. If a caller passes an explicit `data_dir`, search **only** that directory and prefer v2.
2. If `data_dir` is `None` (the default `DATA_DIR` path), prefer v2:
   - First check `DATA_DIR / "updated_connectom_0906_26" / "corrs_mouse_v2.mat"`. If present → return `(path, "v2")`.
   - Else check `DATA_DIR / "corrs_mouse_v2.mat"` (in case the file is later moved up). If present → return `(path, "v2")`.
   - Else `DATA_DIR / "corrs_mouse.mat"`. If present → return `(path, "v1")`.
   - Else `FileNotFoundError` listing all three paths tried.
3. For `species == "human"`, behaviour is unchanged.

#### "What if both v1 and v2 exist?"

v2 wins. Intentional: the v2 file is correct, the v1 file is legacy.

#### Signature change

Change return type from `Path` to `tuple[Path, str]` where the string is `"v1"` or `"v2"`.

**External callers** of `_mat_path` (audit shows one: `pipeline/06_bootstrap.py:42`):

```python
from homer.data import _MAT_TOPKEY, _mat_path, load_cached
```

Either update the caller to unpack, or add a thin wrapper `mat_file_for(species, data_dir=None) -> Path`.

### `load_struct` consequence

Under v2 the top-level dict has more keys (`m.rr`, `m.dirs`, `m.species`, `m.rs4d_size`, `m.version`, `m.ht_info`, `m.info2`, `m.hdr_ABAccf3_2017`, `m.hdr_DSURQE`). No code change needed because the function only returns `d[top]`.

**Hazard:** `mat73.loadmat` will eagerly parse all top-level keys including the two MATLAB nifti header structs. **Assumption to verify:** `mat73.loadmat` on `corrs_mouse_v2.mat` returns without error. If it fails, switch the mouse-v2 path to selective h5py.

---

## 3. `load_metadata` change

The existing logic walks `t_refs[i, j]` for `i in range(n_cols)` and already adapts to any number of columns. **No structural change is needed** for v2: a 19-row cell array is handled the same way as a 7-row cell array.

### One caveat

For v2, the `AS_ix` cell at each node is an `(N, 1)` array of MATLAB linear indices. After `flatten().astype(np.float64)`, it becomes a 1D `float64`. Values fit in `float64` losslessly. The conversion to int64 and the 1-based→0-based subtraction is `parse_t_table`'s responsibility.

### Optional logging

```python
out["_schema"] = _detect_schema(out["ht"])    # "v1" or "v2"
```

### No other change

`stream_mean_fc` and friends only touch `m.rr`. Bit-identical to v1; zero changes.

---

## 4. `parse_t_table` v2 path

This is the substantive change. Branch on `_detect_schema(ht)`. The v1 path stays exactly as is.

### v2 path: exact field-by-field decoding

| ht idx | ht name                        | row idx in `t` | source dtype after `load_metadata` | parse-side dtype           | shape          | semantics                                                                                                              |
|--------|--------------------------------|----------------|------------------------------------|----------------------------|----------------|------------------------------------------------------------------------------------------------------------------------|
| 1      | `type`                         | `row[0]`       | `float` scalar                     | `int8`                     | scalar         | unchanged from v1                                                                                                      |
| 2      | `numid`                        | `row[1]`       | `float` scalar                     | `int32`                    | scalar         | unchanged from v1                                                                                                      |
| 3      | `pairid`                       | `row[2]`       | `float` scalar                     | `int32`                    | scalar         | unchanged from v1                                                                                                      |
| 4      | `region`                       | `row[3]`       | `str`                              | `object` (str)             | scalar         | unchanged from v1                                                                                                      |
| 5      | `subregion`                    | `row[4]`       | `str`                              | `object` (str)             | scalar         | unchanged from v1                                                                                                      |
| 6      | `AS_center_mm`                 | `row[5]`       | `(3,) float64`                     | `float64`                  | (3,)           | world mm in NS frame (ABA CCFv3 25 µm, axcodes PIR, affine origin (0,0,0)). Split into 3 scalar columns.              |
| 7      | `AS_ix`                        | `row[6]`       | `(N,) float64` of MATLAB 1-based   | `int64` 1D array           | (N,)           | **converted to 0-based at load time**                                                                                  |
| 8      | `AS_center_ix`                 | `row[7]`       | `float` scalar (MATLAB 1-based)    | `int64` scalar             | scalar         | **converted to 0-based at load time**                                                                                  |
| 9-12   | `AS_region_*`                  | `row[8-11]`    | `str`                              | `object` (str)             | scalar         | Allen full NAMES (not acronyms)                                                                                        |
| 13     | `DS_center_mm`                 | `row[12]`      | `(3,) float64`                     | `float64`                  | (3,)           | world mm in SS frame (DSURQE 70 µm, axcodes RAS, origin (-6.27, -8.19, -4.20))                                         |
| 14     | `DS_ix`                        | `row[13]`      | `(N,) float64` of MATLAB 1-based   | `int64` 1D array           | (N,)           | **converted to 0-based at load time**                                                                                  |
| 15     | `DS_center_ix`                 | `row[14]`      | `float` scalar (MATLAB 1-based)    | `int64` scalar             | scalar         | **converted to 0-based at load time**                                                                                  |
| 16-19  | `DS_region_*`                  | `row[15-18]`   | `str`                              | `object` (str)             | scalar         |                                                                                                                        |

### Index convention — explicit rules

**For all four fields `AS_ix`, `AS_center_ix`, `DS_ix`, `DS_center_ix`:**

1. The source is a MATLAB 1-based linear index, Fortran (column-major) ordered.
2. `parse_t_table` converts to **0-based** at load time: `idx_0based = idx_matlab.astype(np.int64) - 1`.
3. After conversion, all `*_ix` values are `int64`. Hard `ValueError` if any post-decrement index is < 0 or ≥ grid_size.
4. **Order convention lives in consumers.** Downstream code calls:
   ```
   np.unravel_index(idx, shape=(528, 320, 456), order='F')   # for NS / AS_ix
   np.unravel_index(idx, shape=(181, 274, 139), order='F')   # for SS / DS_ix
   ```
5. **Grid shapes are loader-internal constants.** Define `_NS_SHAPE = (528, 320, 456)` and `_SS_SHAPE = (181, 274, 139)` at module top.

### Decision: linear vs decoded ijk

**Keep linear (1D), but 0-based.** Pre-applying the 1-based correction at load makes the "subtract 1 before unravel" landmine impossible to forget.

### Resulting v2 DataFrame columns (in this exact order)

| column                          | dtype                  | meaning                                                                                          |
|---------------------------------|------------------------|--------------------------------------------------------------------------------------------------|
| `type`                          | `int8`                 | unchanged                                                                                        |
| `numid`                         | `int32`                | unchanged                                                                                        |
| `pairid`                        | `int32`                | unchanged                                                                                        |
| `region`                        | `object` (str)         | unchanged                                                                                        |
| `subregion`                     | `object` (str)         | unchanged                                                                                        |
| `x`                             | `float64`              | **= `DS_center_mm[0]`** — SS world mm (backward-compatibility column)                            |
| `y`                             | `float64`              | **= `DS_center_mm[1]`** — SS world mm                                                            |
| `z`                             | `float64`              | **= `DS_center_mm[2]`** — SS world mm                                                            |
| `hemisphere`                    | `object` (str)         | derived from region prefix (unchanged logic)                                                     |
| `garin_anchor`                  | `bool`                 | `type == 1` (unchanged)                                                                          |
| `anchor_pair_id`                | `Int64`                | unchanged                                                                                        |
| `voxel_indices`                 | `object`               | **= `DS_ix` (0-based, 1D, F-order, into SS grid)**                                               |
| `centre_ns_x/y/z`               | `float64`              | `AS_center_mm[0/1/2]` — NS world mm (axcodes PIR, origin (0,0,0))                                |
| `centre_ss_x/y/z`               | `float64`              | `DS_center_mm[0/1/2]` — SS world mm                                                              |
| `ns_center_ix`                  | `int64`                | scalar 0-based linear index into NS grid, F-order                                                |
| `ss_center_ix`                  | `int64`                | scalar 0-based linear index into SS grid, F-order                                                |
| `ns_voxel_indices`              | `object` (int64 1D)    | 0-based linear, F-order, into NS grid                                                            |
| `ss_voxel_indices`              | `object` (int64 1D)    | 0-based linear, F-order, into SS grid                                                            |
| `region_center_ns_aba`          | `object` (str)         | Allen full NAME at NS centre voxel                                                               |
| `region_center_ns_dsq`          | `object` (str)         | Allen full NAME (DSURQE atlas) at NS centre voxel                                                |
| `region_center_ss_aba`          | `object` (str)         | Allen full NAME at SS centre voxel                                                               |
| `region_center_ss_dsq`          | `object` (str)         | Allen full NAME (DSURQE atlas) at SS centre voxel                                                |
| `region_vote_ns_aba`            | `object` (str)         | majority-vote ABA label over NS voxel set                                                        |
| `region_vote_ns_dsq`            | `object` (str)         | majority-vote DSURQE label over NS voxel set                                                     |
| `region_vote_ss_aba`            | `object` (str)         | majority-vote ABA label over SS voxel set                                                        |
| `region_vote_ss_dsq`            | `object` (str)         | majority-vote DSURQE label over SS voxel set                                                     |

Index: `df.index = df["numid"].astype(int).astype(str)`, index name `"node_id"`.

### Validation gates in `parse_t_table` (v2 path)

1. `len(t) > 0` and every row has exactly 19 cells.
2. `AS_center_mm.shape == (3,)` and `DS_center_mm.shape == (3,)` for every row.
3. After `-1` decrement: every `AS_ix` value in `[0, prod(_NS_SHAPE))`. Every `DS_ix` in `[0, prod(_SS_SHAPE))`. Same for center_ix scalars.
4. `numid` is exactly `1..len(t)`.
5. No region/subregion string is None — only empty.

---

## 5. Backward compatibility — population of v1-era columns

### `x`, `y`, `z`

Populate from `DS_center_mm` (SS frame). Justification: v1 `center` ≈ v2 `DS_center_mm` to sub-voxel precision (mean 0.034 mm, max 0.16 mm). Populating from `AS_center_mm` would silently shift everything by ~6 mm.

### `voxel_indices`

**Populate from `DS_ix` and document the grid-shape change loudly.** The rsmask grid is `(62, 94, 47)`; v2 `DS_ix` is into `(181, 274, 139)`. Pipeline scripts that use `rsmask_shape` to unravel need updating.

**Safety net via `df.attrs`:**

```python
df.attrs["schema"] = "v2"
df.attrs["voxel_indices_grid"] = "SS"
df.attrs["voxel_indices_shape"] = (181, 274, 139)
df.attrs["voxel_indices_order"] = "F"
df.attrs["voxel_indices_one_based"] = False
df.attrs["xyz_frame"] = "SS"
```

`df.attrs` does NOT survive h5ad — see §7 for AnnData equivalent.

---

## 6. New columns and `label_to_allen_id` helper

`centre_ns_*` / `centre_ss_*` uses British spelling to avoid colliding with the v1 `center` ht-column name.

### `label_to_allen_id(name: str) -> int | None`

1. Accepts Allen full NAME as Paul stores it.
2. Tries: exact match, comma-stripped match, re-inserted-commas, case-insensitive.
3. Cache lookup table at module-import time.
4. **Unit test** must enumerate every distinct label string in v2 and assert all resolve.

**Assumption to verify:** allensdk or shipped CSV available for the lookup.

---

## 7. AnnData propagation

### Changes for v2

1. Continue to populate `A.uns["voxel_indices"]` with `DS_ix`.
2. **Additionally** populate:
   - `A.uns["ns_voxel_indices"]` — list of NS-grid linear arrays
   - `A.uns["ss_voxel_indices"]` — list of SS-grid linear arrays (alias for `voxel_indices`)
   - `A.uns["mouse_schema"] = "v2"`
   - `A.uns["ns_grid_shape"] = (528, 320, 456)`, `A.uns["ss_grid_shape"] = (181, 274, 139)`
   - `A.uns["ns_axcodes"] = "PIR"`, `A.uns["ss_axcodes"] = "RAS"`
   - `A.uns["ns_affine_origin"] = (0.0, 0.0, 0.0)`, `A.uns["ss_affine_origin"] = (-6.27, -8.19, -4.20)`

3. New v2 var columns stay in `A.var` (per-node scalars/strings serialise fine).
4. v1-style pop-on-write/restore generalises to handle three ragged keys.

### Cache invalidation

`load_cached` for `species == "mouse"` raises if `A.uns.get("mouse_schema") != "v2"` — error message must include the `rm` command to clear the cache.

---

## 8. Tests

### `tests/test_data_io.py`

Unit tests (no external data):
1. `test_detect_schema_v1` / `_v2` / `_rejects_unknown` / `_rejects_v2_with_extra_column`
2. `test_parse_t_table_v2_synthetic` — synthetic 2-row v2 `t` asserting all 30 columns
3. `test_parse_t_table_v2_rejects_out_of_range_index` / `_zero_index` / `_negative_index`
4. `test_parse_t_table_v1_unchanged`
5. `test_label_to_allen_id_round_trip` / `_handles_paul_comma_stripping`

### `tests/test_data.py`

Integration tests gated on file presence:
6. `test_v2_file_present_and_loadable`
7. `test_v2_indices_decode_to_centres_exactly` — NS frame, `atol=1e-6` (locks in the empirical 0.0000 mm result)
8. `test_v2_ss_indices_decode_to_centres` — SS frame, with documented threshold
9. `test_v2_xyz_compat_with_v1` — `|v2.x - v1.x|.max() < 0.2 mm`
10. `test_v2_rr_unchanged` — sampled-subjects bit-identity
11. `test_v2_anndata_uns_keys`
12. `test_v2_cache_invalidates_v1`

Update existing tests:
- `test_parse_t_table_rejects_wrong_header` — message anchor
- `test_data_dir_present` — accept either v1 or v2 location
- `test_voxel_indices_present_and_nonempty` — add SS-grid range assertion

---

## 9. Migration plan for downstream callers

### Files that need code changes for v2

Read `voxel_indices` and unravel into `rsmask.shape`. Under v2 must switch to SS grid + SS affine OR use `ns_voxel_indices` + NS affine (cleaner):

- `pipeline/00_external/_mouse_transform.py`
- `pipeline/00_external/01_mouse_sc.py:186`
- `pipeline/00_external/02_mouse_genes.py:177-191`
- `pipeline/00_external/02b_mouse_genes_direct.py:271-285`
- `pipeline/00_external/00_inspect_masks.py`
- `pipeline/00_external/warp_rebuild/01_parcel_ccf_labels.py` and `04_warped_voxel_sets.py` — pin to v1 as archival
- `experiments/autism_subtypes/allen_expansion/download_pagani_ish.py:182-186`
- `pipeline/00_external/00b_verify_alignment.py:99`
- `pipeline/00_external/00c_align_mouse_to_ccf.py:70` — deprecate

### Files that work unchanged under v2

Read `x/y/z` (SS frame, same as v1) or `region`/`pairid`:

- `src/homer/data/anchors.py` / `region_anchors.py` / `supplementary_anchors.py` / `atlas_regions.py`
- `src/homer/data/anchor_packs/_dsurqe.py` (47, 139) — explicitly DSURQE-frame
- `src/homer/viz/notebook.py`, `viewer.py`, `gui.py`
- All `pipeline/05*.py` through `08*.py` — use `load_cached` + xyz only
- All `experiments/anchor_packs/*.py`

### Human side: `03_human_sc.py`, `04_human_genes.py` — unaffected.

---

## 10. Off-by-one / grid-confusion landmines

| ID | Wrong call | Prevention |
|----|-----------|------------|
| L1 | Forgetting to subtract 1 before unravel | Loader subtracts at parse; `assert ix.min() >= 0`; tests #6-#8 |
| L2 | Double-applying 1-based correction | `df.attrs[...]`, `A.uns["mouse_schema"]`; per-script docs |
| L3 | C-order instead of F-order | `df.attrs["voxel_indices_order"] = "F"`; test #13 catches |
| L4 | Mixing NS and SS grids | Distinct column names; bounds check (NS goes to 77M, SS to 6.9M); helper `_unravel(ad, frame, idx)` |
| L5 | Mixing NS and SS affines | Load affine from shipped templates by frame; `A.uns["{ns,ss}_axcodes"]` |
| L6 | Confusing v2 `x/y/z` (SS) with NS world | Doc: "`x/y/z` are in SS frame"; `A.uns["xyz_frame"] = "SS"` |
| L7 | Treating v2 `voxel_indices` as rsmask grid | Bounds check (SS values reach 6.9M, rsmask has 273k); `df.attrs["voxel_indices_shape"]` |
| L8 | Treating v2 strings as acronyms | Doc: "Allen full NAMES"; `label_to_allen_id` helper; test #11 |
| L9 | Caching v1 build, loading under v2 expectations | `load_cached` raises on schema mismatch |
| L10 | `_mat_path` returning wrong file | v2-first precedence; log file + schema |
| L11 | `mat73.loadmat` choking on nifti headers | Smoke test first; fallback h5py selective load |
| L12 | `ht` decoding | Already h5py-based, size-agnostic; safe |
| L13 | Detect by `m.version` | Forbidden; `_detect_schema` only takes `ht` |
| L14 | `numid` reordering | Test #15 anchors row-by-row identity v1↔v2 |
| L15 | `df.attrs` lost on h5ad write | Mirror to `A.uns` |

---

## Sequencing for implementation

1. Verify L11: `mat73.loadmat` parses v2 cleanly.
2. Add constants: `_V1_HT`, `_V2_HT`, `_NS_SHAPE`, `_SS_SHAPE`.
3. Add `_detect_schema(ht)`.
4. Update `_mat_path` (tuple return; v2-first precedence). Update `06_bootstrap.py`.
5. Add `load_metadata` `_schema` key.
6. Branch `parse_t_table` on `_detect_schema(ht)`.
7. Add `label_to_allen_id`.
8. Update `build_anndata` to populate v2 `A.uns` keys.
9. Update `load_cached` to refuse v1 cache when v2 .mat present.
10. Add unit tests; run.
11. Add integration tests; run.
12. Full existing test suite for regression.
13. Smoke-build a v2 AnnData.

---

## Open assumptions (flagged, not derived)

1. `corrs_human.mat` still has the v1 7-column `ht`.
2. `mat73.loadmat` parses `corrs_mouse_v2.mat` cleanly including the nifti header structs.
3. We have access to a canonical Allen full-name → Structure ID table.
4. SS affine derived from `template_ABA_SS.nii.gz` matches the affine implied by `DS_center_mm`/`DS_ix` decoding within 0.05 mm at every node.
5. `m.rr` is shape `(105, 1864, 1864)` and bit-identical to v1's — already verified.
6. No script outside the ones audited reads the literal column name `"center"`.
