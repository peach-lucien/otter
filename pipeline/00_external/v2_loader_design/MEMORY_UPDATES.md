# Memory updates for the v2 baseline

The sandbox can't write to the memory folder directly. Three updates to
apply manually:

## 1. Create `homer-v2-baseline.md`

```markdown
---
name: homer-v2-baseline
description: HOMER v2 production baseline as of 2026-06-09 — schema, validation numbers, code paths, and pending follow-ups
metadata:
  type: project
---

The HOMER project switched to the **v2 mouse coordinate package**
(`corrs_mouse_v2.mat`) on 2026-06-09. See [[homer-project]] and
[[homer-cleanup-handoff]] for prior state.

**Why:** the v1 mouse coordinates depended on a heuristic 48-permutation +
centroid-translation transform that mislocated parcel centres by an
average of 4.3 mm vs Paul's nonlinear DSURQE→CCFv3 warp. v2 ships the
pre-warped voxel indices inside the .mat file so no coordinate transform
is needed at load time.

**How to apply:** the v2 loader (`homer.data.io.parse_t_table` v2 branch)
exposes 30 columns including the legacy `x/y/z`/`voxel_indices` (now in
the SS DSURQE frame) plus new `centre_ns_*`, `ns_voxel_indices`,
`region_center_*_aba`/`*_dsq`, `region_vote_*_aba`/`*_dsq` columns.
Downstream code that consumed `voxel_indices` against the v1 rsmask grid
(62×94×47) MUST migrate to either `ss_voxel_indices` (181×274×139
DSURQE grid) or `ns_voxel_indices` (528×320×456 CCFv3 grid).
`_mouse_transform.py` now raises loudly on out-of-bounds rather than
silently dropping voxels.

**Validation deltas (v1 → v2):**

- Beauchamp top-1: 39.2 % → **45.7 %** (+6.5 pp; enrichment 50.6× over null)
- Fulcher Panel 1: r=+0.230 → **+0.373** (+0.143)
- Pagani gene-spatial: r=+0.548 → **+0.601** (Spearman ρ=+0.643)
- TransBrain cortex top-3: 35 % → **41 %** (centroid 25.3 mm vs 39.8 mm null)
- Bootstrap argmax stability: ~0.94 → **0.982**
- Margulies parcel \|r\|: 0.408 → 0.402 (within noise)
- BICCN Th: +0.086 → +0.080 (within noise)
- Balsters dlPFC mass: 0 % → 0 % (falsification holds)
- Buckner association-gap: 6.6 → 6.7 log units (falsification holds, p=3.4e-7)
- Trust anchored-and-validated: 31 % → 31 % (587 parcels)

**Migration docs:** `pipeline/00_external/v2_loader_design/{SPEC.md,
REVIEW.md, RUNBOOK.md}`.

**Production v2 scripts:** `02c_mouse_genes_v2.py`,
`01b_mouse_sc_v2.py`, `diff_anchor_packs_v1_vs_v2.py`.

**v1 paths retained** for the human side (`corrs_human.mat` still v1)
and v1 reproductions, with deprecation banners. `warp_rebuild/` is
archival.

**Tests:** 22 new in `tests/test_v2_loader.py`. NS round-trip tolerance
1e-9 mm; SS 0.8 mm (`DS_center_mm` is a continuous COM not a voxel
centre — `A.uns["ss_center_voxel_is_com"] = False`).

**Pending: option (c) `_dsurqe.py` refactor.** Paul's 8 pre-computed
label columns are LOADED but NOT CONSUMED. Anchor packs still use the
live `mouse_parcels_in_dsurqe_region` via the Beauchamp 2022 DSURQE
label volume + hand-calibrated `DSURQE_OFFSET_MM`. Live-vs-Paul agree
on ~97% of parcels (1810/1864). Gate before swap:
`pipeline/00_external/diff_live_vs_paul_labels.py`. Run after the v2
docs+notebook sweep stabilises.
```

## 2. Update `homer-project.md`

Append at the end of the body:

> **As of 2026-06-09**, the project shifted to v2 mouse coordinates
> (Paul's nonlinear DSURQE→CCFv3 warp via `corrs_mouse_v2.mat`). The v2
> cascade improved Beauchamp top-1 from 39.2 % → 45.7 %, Fulcher Panel 1
> from +0.230 → +0.373, Pagani gene-spatial from +0.548 → +0.601,
> TransBrain cortex top-3 from 35 % → 41 %. Bootstrap stability went
> from ~0.94 to 0.982. Both falsification tests (Balsters, Buckner)
> still pass under v2. See [[homer-v2-baseline]].

## 3. Append to `homer-cleanup-handoff.md`

Add a section at the END (preserve everything above):

```markdown
**v2 migration completed 2026-06-09:**

- v2 loader shipped in `src/homer/data/io.py` (schema-branched on `ht`
  exact equality; MATLAB 1-based F-order indices converted to 0-based
  at load time).
- New migration scripts: `pipeline/00_external/02c_mouse_genes_v2.py`,
  `01b_mouse_sc_v2.py`.
- Production paths now read `ns_voxel_indices` / `ns_center_ix` directly
  from the v2 t-table — no coordinate-transform step required.
- v1 paths carry deprecation banners but remain runnable for v1
  reproductions.
- `pipeline/00_external/warp_rebuild/` retired as archival.
- Adversarial-review docs at
  `pipeline/00_external/v2_loader_design/{SPEC.md, REVIEW.md, RUNBOOK.md}`.
- 22 new v2 loader tests added in `tests/test_v2_loader.py` (all passing).
- Validation cascade re-run; headline numbers improved or stable, both
  falsification tests still pass. See [[homer-v2-baseline]] for the
  full v1 → v2 delta table.

**Open: `_dsurqe.py` option (c) refactor.** Paul's 8 pre-computed label
columns are loaded into `A.var` but not consumed by anchor packs or
Beauchamp validation, which still use the live atlas lookup. Diff
diagnostic written at `pipeline/00_external/diff_live_vs_paul_labels.py`
— run it after this docs+notebook sweep stabilises; if Jaccard ≥ 0.95
on every pack, do the swap.
```

## 4. Add to `MEMORY.md` index

Insert the line below the existing HOMER entries:

```
- [HOMER v2 baseline](homer-v2-baseline.md) — v2 mouse coordinate package, validation deltas, code paths, pending _dsurqe.py refactor
```
