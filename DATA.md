# Data and artifacts

The published archive is v1.3.0 and contains `pi_canonical.npy`. See the
`_canonical_coupling_note` in `data_manifest.json`.

The OTTER code lives in this Git repository. The data and generated artifacts do
**not**. They are too large for Git, and most of the inputs are third-party data
we are not the right party to redistribute. They are hosted as a versioned archive
with a citable DOI, and fetched with `scripts/fetch_data.py`.

```bash
python scripts/fetch_data.py            # reproduce bundle (default)
python scripts/fetch_data.py --tier raw # add the full raw inputs (for a from-scratch rebuild)
```

The DOI and download URLs are read from `data_manifest.json`.
Data DOI: [10.5281/zenodo.20733162](https://doi.org/10.5281/zenodo.20733162), which resolves to
the latest version. Both archives are on that record.

## Available without a download

- Run the unit test suite. `tests/conftest.py` builds a small synthetic
  cross-species problem; data-backed tests `pytest.skip` when their files are
  absent. So `pytest -q` works on a bare checkout.
- Inspect the machine-readable analysis results in `outputs/logs/`
  (the `*.json`/`*.csv` files) are committed to the repo, along with
  `outputs/anndata/_schaefer_order.txt`, the voxel-count arrays and the
  explorer's interface metadata.

Re-running the validations and notebooks, or rebuilding the coupling, requires
the archive below.

## The three tiers

| Tier | Where | Enables |
|---|---|---|
| 0, small artifacts | committed to Git | run unit tests, read all result numbers |
| 1, reproduce bundle | Zenodo `otter-reproduce-v1.3.0.tar.gz` (~735 MB download) | re-run every experiment/notebook against the precomputed couplings |
| 2, raw inputs | Zenodo `otter-raw-inputs-v1.0.0.tar.gz` (606 MB download) | rebuild the coupling bitwise from raw data via `pipeline/` |

Tier 2 is required only to regenerate `π` from scratch, not to use OTTER.

---

## Zenodo record contents

The record holds two archive files. Both are built from the repository root, so
the paths inside each tarball are repo-relative and the fetch script unpacks at
the repository root. The `tar` commands are in
[`scripts/build_archives.sh`](scripts/build_archives.sh).

### Archive 1, `otter-reproduce-v1.3.0.tar.gz` (~735 MB gzipped)

The shipped couplings and inputs needed to run the unrestricted analyses and
notebooks. The QPN-NC validation requires separate authorized access to the
restricted cohort. The exact archive file list is in `scripts/build_archives.sh`
(the `REPRODUCE` array).

**OTTER-generated (we own these, safe to redistribute):**

| Path | Size | What |
|---|---:|---|
| `outputs/coupling/pi_canonical.npy` | 30 MB | the canonical coupling π (1864×2094), what `load_pi()` returns |
| `outputs/coupling/pi_canonical_sharp.npy` | 30 MB | canonical recipe with a later effective solver time, included for concentration sensitivity |
| `outputs/coupling/pi_*.npy` | ~475 MB | analysis-specific ablation and sensitivity couplings |
| `outputs/coupling/trust_multisource_canonical.npz` | ~0.1 MB | interface display metadata used by the explorer; field names are retained for file compatibility |
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
| `data_external/enigma/cortical_thickness_parkinsons_HY*.csv` | <0.1 MB | Four ENIGMA-Parkinson Hoehn–Yahr stage effect-size tables |
| `data_external/_domhof_extracted/` | 1.7 MB | Schaefer-400 + JuBrain atlas NIfTIs |
| `data_external/_diagnostics/` | 0.2 MB | parcellation NIfTI, mask info, mouse→CCF transform |
| `data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc` | 0.3 MB | DSURQE mouse atlas labels |
| `data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/DSURQE_tree.json` | 0.1 MB | DSURQE label tree |
| `data_external/p6ebec-hbp-d000038_SC-FC_HCP_eNKI_pub/Schaefer2018_400Parcels_17Networks.zip` | 7.8 MB | Schaefer parcellation |

### Archive 2, `otter-raw-inputs-v1.0.0.tar.gz` (606 MB gzipped, optional)

The redistributable contents of `data_external/`, for a bitwise rebuild of π
through `pipeline/`. It is a superset of the inputs in Archive 1 plus the full
587 MB `MouseHumanTranscriptomicSimilarity` atlas repo and the small ISH caches.
The Allen ISH download cache (`data_external/_ish_cache/`) is **not** included.
It can be regenerated through the Allen API and is documented in
`pipeline/00_external/`.

Restricted QPN-NC files are never included in either archive. The QPN-NC
validation reads approved downloads from `../data_external/qpn-nc-r01/` and
commits only aggregate statistics without participant identifiers, maps,
scores or small-cell stage counts.

---

## Licensing

The files under `data_external/` are **derived from third-party datasets**, each
with its own terms. Entries whose terms do not permit redistribution are
excluded from the archives, and users re-download them with the scripts in
`pipeline/00_external/`, which fetch from the original sources. The
OTTER-generated artifacts, meaning the couplings, interface metadata, per-disorder
predictions and the processed AnnData caches, carry no redistribution
restriction. Sources and any redistribution restrictions are recorded per dataset in
`pipeline/00_external/README.md` and in the `SOURCES.md` files under
`data_external/*/`.
