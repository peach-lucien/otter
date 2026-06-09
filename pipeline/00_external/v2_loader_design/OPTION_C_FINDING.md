# Option (c) refactor — investigated and deferred

> **Status:** investigated 2026-06-09. Implementation attempted and
> reverted. Documented for future work.

## What option (c) was

Refactor `src/homer/data/anchor_packs/_dsurqe.py` and
`pipeline/05f_beauchamp_validation.py` to consume Paul's pre-computed v2
DSURQE vote labels (`M.var["region_vote_ss_dsq"]`) instead of doing a
live atlas lookup against Beauchamp 2022's
`DSURQE_CCFv3_labels_200um.mnc` volume + the hand-calibrated
`DSURQE_OFFSET_MM = [-0.027, -2.334, +1.018]` constant.

Expected benefits:

- remove the hand-calibrated 6-anchor offset (never properly re-derived)
- drop the Beauchamp 2022 atlas volume as a hard dependency
- make Paul's pre-computed columns load-bearing rather than decorative

## Why it was reverted

**Paul's vote vocabulary uses different region names than the anchor
packs query for.** A naïve subtree-membership check returns empty sets
for most pack queries because the names don't align.

Concrete examples from the v2 file (114 unique vote strings):

| Anchor pack queries for ... | Paul's vote vocabulary has ... |
|---|---|
| `Caudoputamen` | `striatum` (the parent in the DSURQE tree) |
| `Periaqueductal gray` (American) | `periaqueductal grey` (British) |
| `Primary motor area` | `Primary motor cortex` |
| `Lateral visual area` | `Secondary visual cortex,lateral area` |
| `Visual areas` | individual subdivisions only — no umbrella term |
| `Field CA1` | `CA1Or`, `CA1Rad`, `CA1Py` (orientation-specific) |
| `Anterior cingulate area` | `Cingulate cortex,area 24a` etc. |
| `Inferior colliculus` | `colliculus,inferior` (reversed word order) |

Net result: Paul's 114 vote strings DON'T form a strict subset of the
pack-query name space. Some pack queries find no Paul match at all
(would return empty); some Paul votes don't correspond to any pack
query name.

## The earlier "ALL CLEAR" diff diagnostic

The diagnostic at
`pipeline/00_external/diff_live_vs_paul_labels.py` reported ALL CLEAR.
That turned out to be a **false positive**: the pack-builder live-path
calls were failing silently in the user's run (a path-resolution issue
in the diff script, since fixed), so no comparisons were actually
computed. The script's "no flagged entries" exit was misleading —
because the empty list could also mean "no entries were tested" rather
than "all entries pass".

## What would be needed to make (c) work

A name-mapping layer that translates pack-query names to Paul's vote
vocabulary. **Most of this is already buildable in-house.** Of Paul's
114 unique vote strings:

- **23 appear verbatim in `DSURQE_tree.json`** — no mapping needed.
- **77 are bridged via the Beauchamp 2022 CSV**
  (`DSURQE_40micron_R_mapping_long.csv`) which has columns
  `Structure` (Paul-style with left/right prefixes) and `ABI`
  (DSURQE_tree.json name). Strip the side prefix, look up.
- **14 are not in the CSV.** 6 of those are cerebellar lobules that
  HOMER excludes from its parcellation anyway. The remaining 8
  non-cerebellar entries are hand-authored in
  `src/homer/data/anchor_packs/_paul_vote_bridge.py` — see that file
  for the actual mapping and per-entry confidence flags.

The 8 hand-mapped entries cover ~5 % of the 1864 parcels. Each is
flagged with a confidence level (HIGH/MEDIUM/LOW) and a question to
ask Paul if confirmation is needed.

Inventory of Paul's shipped documentation (checked 2026-06-09):

- `data_crossspecies/crossspecies_info.pptx` — table schema only
- `data_crossspecies/check_tables_mouse_human.txt` — actual row data
- `data_crossspecies/warpfields/info.pptx` — warpfield convention
- `data_crossspecies/updated_connectom_0906_26/info_v2.pptx` — v2
  column documentation (the 19 columns), no vocabulary mapping
- `data_crossspecies/_orig_ccfv3_2017/infos.pptx` — Allen NRRD origin

**None of Paul's docs ship a vocabulary mapping**, so the bridge has
to come from us. The Beauchamp CSV + 8 hand-mapped entries gets us to
100% coverage.

Either way, the gating mechanism is:

```python
PACK_NAME_TO_PAUL_VOTES = {
    "Caudoputamen":         {"striatum"},  # ⚠ over-broad — see below
    "Periaqueductal gray":  {"periaqueductal grey"},
    "Primary motor area":   {"Primary motor cortex"},
    "Field CA1":            {"CA1Or", "CA1Rad", "CA1Py"},
    # ...
}
```

with the caveat that Paul's coarser-grained votes (e.g. "striatum"
covers Caudoputamen + NAc + olfactory tubercle in the DSURQE
hierarchy) are **strictly less specific** than the live atlas lookup.
For those regions, the option-(c) refactor would degrade granularity,
not improve it. The right policy is: for the few coarse-vote regions,
keep the live lookup; for the ones where Paul's vocabulary aligns
1:1, swap to columns.

## Tests that lock the finding

`tests/test_dsurqe_v2_dispatch.py`:

- `test_paul_vote_vocabulary_smaller_than_pack_query_vocabulary` —
  asserts that Paul ships ~114 votes and that 6 specific
  pack-query names are NOT in Paul's vocabulary. If a future v2
  Paul-file converges the naming, this test will fail loudly and
  prompt a reconsider.
- `test_dsurqe_offset_constant_still_present` — asserts the live
  path's `DSURQE_OFFSET_MM` constant is still in place, so the v1
  lookup hasn't accidentally been removed.

## Recommendation

Defer option (c) until either:

- A name-mapping table exists (ask Paul, or build one from the DSURQE
  tree by inspecting which leaf names overlap each pack-query region's
  subtree).
- A future v2.x of Paul's table uses an Allen-aligned vote vocabulary.

The current `_dsurqe.py` live lookup is **functionally correct** under
v2 (anchor packs build successfully; Beauchamp validation runs; numbers
match what we reported). The architectural cleanup is desirable but
not urgent.

## What remains in the codebase

- `_build_dsurqe_ancestor_map` and `_has_v2_dsurqe_votes` utilities in
  `_dsurqe.py` — kept as infrastructure for a future option-(c)
  attempt with the name-mapping layer.
- `DSURQE_OFFSET_MM` constant — still load-bearing; do not remove.
- The test file documenting the finding —
  `tests/test_dsurqe_v2_dispatch.py`.
- This document — `pipeline/00_external/v2_loader_design/OPTION_C_FINDING.md`.

## Lesson for future migrations

When pipelines have two sources of an ostensibly-equivalent
computation, the equivalence claim is on the **VALUES**, not on the
**NAMES**. Spot-checking by names — as the original diff diagnostic
did — masks vocabulary mismatches. The right check is to compare the
resulting parcel SETS for each pack-query name, looking at the live
path's output independently and the v2 path's output independently,
and asserting Jaccard ≥ 0.95.

The fixed diff diagnostic (`diff_live_vs_paul_labels.py`) now does
this correctly. Future maintainers should re-run it before
re-attempting option (c).
