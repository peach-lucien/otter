# DSURQE parcel-lookup cross-check

This diagnostic compares two mappings from OTTER mouse parcels to DSURQE regions:

- atlas lookup around each parcel centroid, as used by the regional-pack builders;
- the precomputed `region_vote_ss_dsq` label in the mouse parcel table.

The two methods differ in spatial support and label granularity, so their assignments need not be identical. This script is not part of model fitting or inferential analyses.

Run from the repository root:

```bash
PYTHONPATH=src python experiments/dsurqe_lookup_crosscheck/compare_lookup_vs_votes.py
```

The comparison is written to `outputs/logs/lookup_vs_votes.csv`.
