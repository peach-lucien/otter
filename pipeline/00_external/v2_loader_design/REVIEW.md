# Adversarial Review of `corrs_mouse_v2.mat` Loader Spec

> Independent reviewer agent, 2026-06-09. Empirical claims verified against
> the actual v2 mat file and shipped NIfTIs.

## BLOCKERS (must address before implementation)

### B1. `_mat_path` caller audit is incomplete — `eda.py` is missed
- `src/homer/data/eda.py:16` imports `_mat_path` and `_MAT_TOPKEY`.
- `src/homer/data/eda.py:28` calls `p = _mat_path(species, data_dir)` inside `streaming_subject_similarity`.

If spec's tuple-return ships, `eda.py:28` silently breaks (TypeError when tuple → `h5py.File`).

**Fix:** add `eda.py:28` to the migration list, OR (cleaner) avoid the signature change — keep `_mat_path` returning `Path` and add sibling `_mat_path_and_schema(...) -> tuple[Path, str]` for internal use.

### B2. SS centre round-trip threshold needs to be ~0.55 mm
Empirical (1864/1864 nodes verified):
- NS frame: `|world(AS_center_ix_0based) − AS_center_mm|.max() == 0.0000` (exact).
- SS frame: max = **0.5495 mm** on 12 nodes (0.64%); everywhere else exact 0.

Cause is upstream-semantic, not loader: `DS_center_mm` is a continuous COM/warped centroid, `DS_center_ix` is the closest member of the voxel set. The chosen voxel is offset from the COM by up to 0.55 mm.

**Fix:**
- Spec §8 test #14: pin `atol = 0.6`.
- Spec §10 add landmine **L16**: "`DS_center_ix` decodes to world coord NOT equal to `DS_center_mm`; max 0.55 mm. Treat `ss_center_ix` as a representative voxel, not the COM."
- Spec §7 add `A.uns["ss_center_voxel_is_com"] = False` (and `True` for NS).

### B3. `_dsurqe.py` small-radius lookups WILL shift under v2
`src/homer/data/anchor_packs/_dsurqe.py:139` `mouse_parcels_in_mouse_sphere` uses `radius_mm` as small as 0.5 mm (e.g., LC, raphe lookups).

Empirical: `|v1.center − v2.DS_center_mm|.max() = 0.117 mm`. At 0.5 mm radius, a 0.12 mm shift on every parcel can change anchor-pack membership at the boundary.

**Fix:** §9 add verification step — for each anchor pack that uses `mouse_parcels_in_mouse_sphere`, build df_v1 and df_v2, compare selected node ID sets, escalate any difference. Don't ship until deltas documented.

### B4. `load_cached` cache-invalidation logic is broken
Spec §7 says "raise if `A.uns.get("mouse_schema") != "v2"`" — but:
- Existing `load_cached` (`io.py:319-327`) doesn't take `data_dir`, so can't resolve the .mat file to know the expected schema.
- Phrasing "≠ v2" is brittle for future v3.
- Behaviour for absent `mouse_schema` key (legacy cache) works only by accident.

**Fix:** spec the comparison as `expected_schema = _detect_schema_from_disk(species, data_dir); if A.uns.get("mouse_schema") != expected_schema: raise CacheStaleError(expected, actual)`. Update `load_cached` signature to take `data_dir`. Include both schemas + `rm` command in error.

### B5. `colleague_voxel_to_ccf_world` silently filters v2 voxels to empty
`pipeline/00_external/_mouse_transform.py:54`:
```python
valid = (idx >= 0) & (idx < int(np.prod(rsmask_shape)))
idx = idx[valid]
```
`rsmask_shape = (62, 94, 47)`, prod = 273,916. Empirical: 0/77,969 v2 `DS_ix` values fall below that bound (min observed = 1,014,278). So `valid.sum() == 0` and downstream gets empty arrays — `02_mouse_genes.py` would silently write a NaN-filled gene matrix.

**Fix:**
- Patch `_mouse_transform.py:54-55` to `raise ValueError` instead of silent-filter, **before** any consumer migration.
- §10 update L7 prevention text accordingly.

### B6. v2 `*_DSURQUE` columns are NOT Allen-ontology strings
Empirical: `AS_region_center_DSURQUE` contains strings like `'CA1Or'`, `'CA1Py'`, `'CA1Rad'`, `'Accessory olfactory bulb,glomerular,external plexiform and mitral cell layer'`. These are **DSURQE atlas** labels, not Allen labels. `label_to_allen_id` on them would return None silently.

**Fix:**
- §4 table line 138: "atlas-specific full names: cols `*_DSURQUE` are DSURQE labels, cols `*_ABA` are ABA labels."
- §6 split helper: `aba_label_to_allen_id(name)` and `dsurqe_label_to_id(name)` (or restrict helper to ABA columns and assert).
- §8 test #11 must enumerate the ABA-only columns.

## IMPORTANT

### I1. Single-voxel parcel would crash `parse_t_table` v2 path
`load_metadata` (`io.py:90-92`) casts shape-`(1,)` arrays to Python `float`. A future v2 parcel with a 1-element `AS_ix` would arrive as `float` not array → `.astype` fails.

Empirical: min `DS_ix` length is 12 today; safe but landmine for future-Paul.

**Fix:** §4 guard `as_ix = np.atleast_1d(np.asarray(row[6], dtype=np.float64)).astype(np.int64).ravel()`. Add unit test.

### I2. NS round-trip test threshold should be `1e-9`, not `1e-6`
Empirical NS max error is **exactly 0.0 mm**. `atol=1e-6` would mask a 0.5 µm regression indicating a unit error.

**Fix:** §8 test #13 use `np.testing.assert_array_equal(...)` or `atol=1e-9`.

### I3. `corrs_human.mat` schema verified (closing open assumption #1)
Verified: human `ht == V1_HT` exactly. Pin in spec as verified, not assumption.

### I4. h5ad serialises tuples as numpy arrays
`A.uns["ns_grid_shape"] = (528, 320, 456)` written as h5py dataset → read back as `np.ndarray`. Equality check `A.uns["ns_grid_shape"] == (528, 320, 456)` returns array, fails in `if`.

**Fix:** §7 store as `np.array([...], dtype=np.int64)` explicitly. Compare via `np.array_equal` or `tuple(arr.tolist())`. Add a write+reload h5ad test.

### I5. `load_struct` fallback for selective h5py undefined
§2 line 98 says switch to selective h5py if mat73 fails. But selective h5py wouldn't populate `m.rr` as numpy array. Sole caller is `tests/test_data.py:19`.

**Fix:** spec either skip `load_struct` for v2 (with the test skipping for v2) OR write a hand-rolled v2 loader returning the dict shape mat73 would produce minus the two MATLAB nifti header structs.

### I6. mat73 unverified on v2 file (open assumption #2)
Sequencing step 1 says verify mat73, but spec review can't run it (sandbox lacks the package). Must be confirmed in the user's env before implementation proceeds.

### I7. Cache `rm` message must list all three cache files
Cache has `{species}.h5ad`, `{species}.fc.npy`, `{species}.voxels.npz` (optional).

**Fix:** error template `rm -f {cache_dir}/{species}.h5ad {cache_dir}/{species}.fc.npy {cache_dir}/{species}.voxels.npz`.

### I8. Pop-restore for ragged keys must handle aliasing
If `voxel_indices` and `ss_voxel_indices` share the same list object, double-pop / double-write may interact badly with anndata's write logic.

**Fix:** §7 explicitly state "distinct copies". Test by writing twice.

### I9. xyz-weight OT cost robustness to 0.117 mm shift unquantified
`pipeline/06_bootstrap.py:55` uses `xyz_weight=0.5`. 0.12 mm shift between v1 and v2 xyz is below the inter-parcel spacing (~1 mm); OT cost should be stable. But the spec asserts "works unchanged" without quantifying.

**Fix:** §9 add: "verify by running OT once on v1 vs v2 and asserting `|cost_v1 − cost_v2| / cost_v1 < 1e-3`."

### I10. `00_inspect_masks.py` migration notes missing
§9 lists the file as "needs code changes" but no line-precise migration. `00_inspect_masks.py:107` calls `parse_t_table` and inspects `voxel_indices` assuming rsmask grid — silent bug under v2.

**Fix:** spec line-precise migration. Either no-op under v2 (loader has already done the inspection work) or adapt to SS grid.

## NICE-TO-HAVE

### N1. `DSURQE` vs `DSURQUE` spelling — Paul uses `DSURQUE`
Spec V2_HT list correctly uses `DSURQUE`. Spec prose has `DSURQE` in places. If an implementer copies from prose, schema detection silently fails.

**Fix:** banner at top of §1: "Paul's spelling is `DSURQUE` (note the extra U). All `*_DSURQUE` strings are case-sensitive."

### N2-N12. Verifications passed
- `numid` is exactly 1..1864 ✓
- `m.version` is a datestring not a version label (`'09-Jun-2026 13:06:18'`) — spec correctly avoids depending on it ✓
- `m.species == 'mouse'` ✓
- `m.rr` shape `(105, 1864, 1864)` float32 — same as v1 ✓
- `m.dirs` shape `(1, 105)` — same as v1 ✓
- `rs4d_size` shape `(4, 105)` — same as v1 ✓
- NS template affine matches §4 claim ✓
- SS template affine matches §4 claim ✓
- 0 / 77,969 v2 `DS_ix` values fall in rsmask bound — strong silent-corruption protection ✓
- All 1864 `*_center_ix` are members of their respective voxel sets ✓
- No comma-stripped collisions in either DSURQUE or ABA label sets ✓
- No None/empty label strings ✓

## Empirical anchors the spec must pin

| Quantity | Empirical value | Where to pin |
|---|---|---|
| NS center round-trip max error | **0.0** mm (exact) | §8 test #13: `atol=1e-9` |
| SS center round-trip max error | **0.5495 mm** (12/1864) | §8 test #14: `atol=0.6` + L16 landmine |
| `\|v1.center − v2.DS_center_mm\|.max()` | **0.117 mm** | §8 test #15: `< 0.2` is fine |
| AS_ix bounds (after −1) | max=73,597,662 < 77,045,760 | confirmed in range |
| DS_ix bounds (after −1) | max=5,678,616 < 6,893,566 | confirmed in range |
| Min array length | AS_ix=294, DS_ix=12 | safe today; I1 guard needed for future |
| 1-based confirmed | no AS_ix or DS_ix value is 0 | safe |
| DS_ix values fitting rsmask bound | **0 / 77,969** | confirms v1 grid not accidentally shareable |

## Top 3 fixes before implementation

1. **B5**: patch `_mouse_transform.py:54` to raise instead of silent-filter — guards against silent gene-matrix corruption.
2. **B2** + **L16**: pin SS test threshold at 0.6 mm and document that `ss_center_ix` is a representative voxel, not the COM.
3. **B1**: include `eda.py:28` in caller migration, or avoid the tuple-return signature change.
