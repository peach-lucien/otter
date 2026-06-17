# Data and artifacts

The HOMER code lives in this Git repository. The data and generated artifacts do
**not** — they are too large for Git, and most of the inputs are third-party data
we are not the right party to redistribute. They are hosted as a versioned archive
with a citable DOI, and fetched with `scripts/fetch_data.py`.

```bash
python scripts/fetch_data.py            # reproduce bundle (default)
python scripts/fetch_data.py --tier raw # add the full raw inputs (for a from-scratch rebuild)
```

The DOI and download URLs are read from `data_manifest.json`.
Data DOI: [10.5281/zenodo.20733163](https://doi.org/10.5281/zenodo.20733163).

## What a fresh clone can do *without* any download

- Run the unit test suite. `tests/conftest.py` builds a small synthetic
  cross-species problem; data-backed tests `pytest.skip` when their files are
  absent. So `pytest -q` works on a bare checkout.
- Read every headline number. The validation result logs in `outputs/logs/`
  (the `*.json`/`*.csv` files) are committed to the repo, along with
  `outputs/anndata/_schaefer_order.txt`, the voxel-count arrays, and the
  multi-source trust map.

To **re-run** the validations and notebooks, or to **rebuild the coupling**, you
need the archive below.

## The three tiers

| Tier | Where | Lets you |
|---|---|---|
| 0 — small artifacts | committed to Git | run unit tests, read all result numbers |
| 1 — reproduce bundle | Zenodo `homer-reproduce-v1.0.0.tar.gz` (173 MB download) | re-run every experiment/notebook against the precomputed coupling |
| 2 — raw inputs | Zenodo `homer-raw-inputs-v1.0.0.tar.gz` (606 MB download) | rebuild the coupling bitwise from raw data via `pipeline/` |

You do **not** need Tier 2 to use HOMER — only to regenerate `π` from scratch.

---

## Zenodo record — exactly what to upload

Create **one Zenodo record** with **two archive files**. Build both from the repo
root so the paths inside the tarball are repo-relative (the fetch script unpacks
at the repo root). The exact `tar` commands are in
[`scripts/build_archives.sh`](scripts/build_archives.sh).

### Archive 1 — `homer-reproduce-v1.0.0.tar.gz` (173 MB gzipped, ~360 MB unpacked)

Everything needed to re-run the experiments and notebooks on the shipped coupling.

**HOMER-generated (we own these — safe to redistribute):**

| Path | Size | What |
|---|---:|---|
| `outputs/coupling/pi_fc_plus_SC_with_all_packs.npy` | 30 MB | the recommended coupling π (1864×2094) |
| `outputs/coupling/trust_multisource_all_packs.npz` | 0.3 MB | per-parcel multi-source trust tiers |
| `outputs/coupling/per_disorder_predictions.npz` | 0.07 MB | ENIGMA per-disorder predicted maps |
| `outputs/anndata/mouse.h5ad` | 42 MB | processed mouse parcel table + features |
| `outputs/anndata/human.h5ad` | 52 MB | processed human parcel table + features |
| `outputs/anndata/mouse.voxels.npz` | 4.6 MB | mouse parcel→voxel index map |
| `outputs/anndata/mouse_voxel_counts.npy`, `human_voxel_counts.npy` | <0.1 MB | per-parcel voxel counts |

**Third-party-derived inputs the validations read (see licensing note):**

| Path | Size | Source |
|---|---:|---|
| `data_external/human_genes.npy` + `human_gene_list.csv` | 125 MB | Allen Human Brain Atlas microarray (derived) |
| `data_external/mouse_genes.npy` + `mouse_gene_list.csv` | 0.5 MB | Allen Mouse Brain ISH (derived) |
| `data_external/human_sc.npy` + `human_sc_meta.json` | 17 MB | HCP/eNKI structural connectivity (Domhof) |
| `data_external/mouse_sc.npy` + `mouse_sc_meta.json` | 14 MB | Allen mouse connectivity |
| `data_external/mouse_sc_knox_augmented.npy`, `knox_sc/` | 28 MB | Knox 2019 mouse connectome (derived + raw CSVs) |
| `data_external/orthologs.csv` + `orthologs_meta.json` | <0.1 MB | mouse↔human gene orthologs |
| `data_external/fulcher_2019_gradients/` | 11 MB | Fulcher 2019 (PNAS) cortical maps |
| `data_external/pagani_2026/` | 1.9 MB | Pagani 2026 (Nat Neurosci) autism subtypes |
| `data_external/transbrain_2025/` | 0.6 MB | TransBrain 2025 benchmark tables |
| `data_external/_domhof_extracted/` | 1.7 MB | Schaefer-400 + JuBrain atlas NIfTIs |
| `data_external/_diagnostics/` | 0.2 MB | parcellation NIfTI, mask info, mouse→CCF transform |
| `data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc` | 0.3 MB | DSURQE mouse atlas labels |
| `data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/DSURQE_tree.json` | 0.1 MB | DSURQE label tree |
| `data_external/p6ebec-hbp-d000038_SC-FC_HCP_eNKI_pub/Schaefer2018_400Parcels_17Networks.zip` | 7.8 MB | Schaefer parcellation |

### Archive 2 — `homer-raw-inputs-v1.0.0.tar.gz` (606 MB gzipped, optional)

The complete `data_external/` directory, for a bitwise rebuild of π through
`pipeline/`. It is a superset of the inputs in Archive 1 plus the full
587 MB `MouseHumanTranscriptomicSimilarity` atlas repo and the small ISH caches.
The Allen ISH download cache for the gene-expansion experiment
(`experiments/.../pagani_ish_cache/`, ~584 MB) is **not** included — it is
regeneratable via the Allen API (slow, 1–3 days) and documented in
`pipeline/00_external/`.

---

## Licensing — read before you publish the archives

The files under `data_external/` are **derived from third-party datasets**, each
with its own terms. Before making the archives public, confirm you may
redistribute them; where you can't, drop those entries from Archive 1/2 and let
users re-download via the scripts in `pipeline/00_external/`, which fetch from the
original sources. The HOMER-generated artifacts (the coupling, trust map,
per-disorder predictions, and the processed AnnData caches) are ours to share
freely. Sources and any redistribution restrictions are recorded per dataset in
`pipeline/00_external/README.md` and in the `SOURCES.md` files under
`data_external/*/`.

## After uploading

1. Reserve/publish the DOI on Zenodo.
2. Put the DOI and the two file URLs into `data_manifest.json`.
3. Paste each file's checksum (Zenodo shows MD5 per file) into the same manifest
   so `scripts/fetch_data.py` can verify downloads.
4. Add the DOI badge to `README.md` (placeholder already in place).
