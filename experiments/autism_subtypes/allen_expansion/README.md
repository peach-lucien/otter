# Allen ISH expansion for Pagani 2026 Test 3

> **Source-data-dependent (not in the public release).** `download_pagani_ish.py`
> needs the mouse resting-state mask `data_crossspecies/_mouse_mask/rsmask.nii`
> (raw source, not shipped in the Zenodo bundles) and performs a multi-day Allen
> API download. It exits with a clear message if the mask is absent. This is a
> maintainer/source-data-only script; the per-parcel result it produces
> (`pagani_mouse_expr.npy`) is already committed so downstream steps don't need it.

Standalone pipeline to expand HOMER's mouse Allen ISH atlas from 61 curated genes to **all 6,415 genes implicated by Pagani et al. 2026** (1,952 hypo-only + 4,463 hyper-only from MOESM4). Required to power Test 3 (gene-spatial translation) properly.

## Why this exists

The in-conversation proof-of-concept used HOMER's 51-gene panel, of which only 36 overlap with Pagani's lists. Spearman ρ = +0.619 (empirical p = 0.045) is suggestive but underpowered. With the full 6,415-gene panel, or however much of it Allen has usable ISH data for (expected 30–50%). Test 3 becomes a properly-powered cross-species spatial replication of Pagani's pathway claim.

## Why this isn't run in-conversation

- **Disk**: each ISH zip is 1–2 MB, so 6,000 genes ≈ 6–12 GB cache. The conversation sandbox has ~20 MB available.
- **Time**: even with 4 parallel workers, Allen's API rate-limits aggressively; realistic wall-clock 1–3 days for the full pull.
- **Dependencies**: `allensdk` is heavy (~50 MB install) and fails in disk-constrained environments. This script uses raw `requests` to bypass it.

So this directory contains a portable script you run on your own machine.

## What's here

| File | What |
|---|---|
| `pagani_gene_list.csv` | 6,415 rows: `human_symbol`, `mouse_symbol`, `subtype` (hypo / hyper / both), composed from MOESM4 sheet `subtypes` |
| `download_pagani_ish.py` | Standalone Allen ISH downloader. No allensdk dependency; just `requests` + `nibabel` + `numpy`. Idempotent (skips cached zips). Saves `pagani_mouse_expr.npy` + `pagani_gene_list_resolved.csv` |
| `run_pagani_gene_test.py` | Rerun Test 3 (gene-spatial translation) using the expanded matrix. Same logic as `09_gene_spatial_translation.py` but with full coverage |
| `README.md` | This file |

## How to run

### One-time setup

```bash
# From HOMER repo root
pip install requests nibabel numpy pandas scipy openpyxl
```

The script imports HOMER's existing transform helpers (`_mouse_transform.py`) and requires:
- `data_external/_diagnostics/mouse_to_ccf_transform.json` (exists)
- `data_crossspecies/_mouse_mask/rsmask.nii` (HOMER's mouse mask, exists)

### Step 1, small test (optional but recommended)

Validate the pipeline on 30 Pagani genes first to confirm the API + parsing chain works on your machine:

```bash
cd <homer-repo-root>
python experiments/autism_subtypes/allen_expansion/download_pagani_ish.py \
    --max-genes 30 --workers 2
```

Expected: completes in 2–5 minutes, ~30 MB cache, ~50–80% success rate.

### Step 2, full pull

```bash
python experiments/autism_subtypes/allen_expansion/download_pagani_ish.py \
    --max-genes 6415 --workers 4
```

Expected: 1–3 days wall-clock, 6–12 GB cache, ~30–50% yield. The script is idempotent, kill and resume freely; cached zips are reused.

To run in chunks across machines:
```bash
# Machine 1
python ... --start 0 --max-genes 2000

# Machine 2 (different output dir or merged cache)
python ... --start 2000 --max-genes 2000
```

### Step 3, run the powered Test 3

```bash
PYTHONPATH=src python experiments/autism_subtypes/allen_expansion/run_pagani_gene_test.py
```

Writes `outputs/logs/autism_subtypes_gene_spatial_expanded.json`. With ~2,000+ genes in each subtype, n=8 networks is no longer the limit; the predicted-vs-observed correlation now has tight standard error.

## Expected outcome

If Pagani's claim 4 is correct AND HOMER's π carries cross-species spatial signal:
- Pearson r should be substantially above the underpowered +0.44 we got with 36 genes
- Empirical p should drop well below 0.045 (Spearman) / 0.13 (Pearson)
- Same-sign agreement should rise from 4/8 toward 6-8/8

If the result *doesn't* improve, that would suggest the cross-species pathway signal is real for the small marker-gene subset but not for the broader Pagani gene lists, informative either way.

## Allen API notes

- Endpoint: `http://api.brain-map.org/grid_data/download/<sds_id>?include=energy`
- Rate-limit: nominal soft limit ~1 req/sec; aggressive in practice. Use `--workers ≤ 4`.
- ~30% of mouse gene symbols have no coronal SectionDataSet (gene not in atlas).
- ~20% of remaining attempts fail the download (server intermittent issues).
- Net yield from a curated gene list is typically 50–70%; from a research gene list like Pagani's, expect 30–50%.

## After Pagani's gene list is downloaded once

The cache is reusable across HOMER experiments. Future studies that need parcel-level mouse expression for any subset of these genes can read from `pagani_mouse_expr.npy` directly without re-querying Allen.
