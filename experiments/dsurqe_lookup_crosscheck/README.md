# DSURQE lookup vs. precomputed votes (maintainer cross-check)

A small diagnostic that quantifies the two ways of answering *"which mouse
parcels sit in DSURQE region R?"*:

- **Live atlas lookup** (production default), place each parcel's centroid
  into the Beauchamp DSURQE label volume and read the majority label in a
  ~1 mm neighbourhood (`src/homer/data/anchor_packs/_dsurqe.py`).
- **Precomputed votes**, the per-parcel `region_vote_ss_dsq` label that ships
  in the mouse table (majority over the parcel's full voxel set), resolved to
  tree names via the Beauchamp CSV + the hand-authored `_paul_vote_bridge`.

This is **not** a user-facing notebook; it exists to justify the default.

## Run

```bash
PYTHONPATH=src python experiments/dsurqe_lookup_crosscheck/compare_lookup_vs_votes.py
```

Self-contained (numpy / h5py / nibabel only); writes `outputs/logs/lookup_vs_votes.csv`.

## What it shows (and why the live lookup is the default)

Per anchor-pack region query it reports `|live|`, `|vote|`, intersection and
Jaccard. The pattern is consistent:

- **Fine sub-region queries collapse to Jaccard 0**, e.g. *Caudoputamen*,
  *barrel field*, *lateral visual area*, *retrosplenial / prelimbic area*. The
  votes only resolve to a coarser parent (e.g. "striatum"), so they cannot
  select the specific region at all.
- **Where the vote vocabulary matches the query granularity**, Jaccard is
  ~0.6–0.75 (motor, auditory, piriform, entorhinal, perirhinal, secondary
  motor, cortical subplate), substantial agreement, not identical (the live
  lookup uses the centroid; the vote uses the full voxel set).
- **Only ~74 % of votes resolve to a tree node** even with comma/case-tolerant
  matching; the rest use names the bridge doesn't cover.

Net: roughly **half** the anchor-pack queries are at a granularity the votes
cannot reach, so the live atlas lookup stays the default. The votes are best
used as a cross-check, or as a selective fallback only where a vote name lines
up 1:1 with a pack query, which would also gain the full-voxel-set robustness.
