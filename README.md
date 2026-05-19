# HOMER

**Hom**ology **E**stimation across species via **R**egional optimal transport.

A Python package that learns soft cross-species correspondences between mouse and human brain parcels using **Fused Gromov–Wasserstein optimal transport**, anchored on published homologue pairs.

Output: a coupling matrix **π** of shape (1864 mouse parcels × 2094 human parcels) where `π[i, j]` is the model's estimated probability that mouse parcel *i* corresponds to human parcel *j*.

---

## Try it in your browser (no install)

**→ [Open the HOMER Mapping Explorer](https://peach-lucien.github.io/homer/)**

A self-contained 3D viewer for the production coupling: search a mouse region or parcel, see its top-K human partners ranked by coupling mass, toggle the cortical surface or mouse atlas shell, and inspect the trust evidence behind every prediction. No Python, no install, no backend — it's a single HTML file with the recommended model baked in. Use this if you want to *look at* HOMER. The rest of the README is for using it programmatically or reproducing it.

---

## Headline numbers

| Metric (recommended `pi_fc_plus_SC_with_all_packs.npy`) | Value |
|---|---:|
| Beauchamp 2022 anchor-overlapping top-1 | **37 %** (3.1× over the strict baseline) |
| Beauchamp top-5 | **46 %** |
| Mean rank of correct human partner / 2094 | **85** (top 4 %) |
| Region-level qualified top-3 (Beauchamp-22 candidate set) | **100 %** |
| Bootstrap argmax stability (40 subject resamples) | **97.8 %** |
| Pagani 2026 name-based network bridge — independent check | **4 / 8** canonical pairs diagonal-argmax (1.92× over null; chance: 2/8, 0.97×). *Methodological check on their scaffolding, not a replication of their findings.* |
| Pagani 2026 subtype-contrast spatial pattern (their claim 3) | **Pearson r = +0.547** on per-network row-sums (n=8). Sharper full-matrix version: **r = +0.537 over 36 network-pair entries, analytical p = 0.0007, empirical p < 0.005** vs permuted-π null. |
| Pagani 2026 gene-set spatial pattern (their claim 4) | Spearman ρ = +0.619, empirical p = 0.045 (proof-of-concept; 36 of Pagani's 6,415 genes covered by HOMER's curated Allen-ISH set). |

Multi-source trust map says: 19 % of mouse parcels are in the `anchored_and_validated` tier, 36 % in `validated_only`, 13 % in `structural`, 29 % `low_evidence`. See [`docs/03_results.md`](docs/03_results.md).

## Install + quickstart (for programmatic use)

Skip this section if you just want to look at the couplings — use the [explorer](https://peach-lucien.github.io/homer/) above. Install only if you need to query π in code, re-train the model, or extend it.

```bash
git clone <this-repo> homer && cd homer
conda env create -f env.yml && conda activate homer
pip install -e ".[dev]"
pytest -q                            # ~10s, 161 tests
```

Query a region after installation:

```python
import numpy as np
from homer.data import load_cached

M, _ = load_cached("mouse", cache_dir="outputs/anndata")
H, _ = load_cached("human", cache_dir="outputs/anndata")
pi = np.load("outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")    # (1864, 2094)

# Top-5 human partners for mouse parcel 1234
top5_idx = pi[1234].argsort()[::-1][:5]
top5_regions = H.var.iloc[top5_idx][["region", "x", "y", "z"]]
print(top5_regions)

# Filter by trust tier
trust = np.load("outputs/coupling/trust_multisource_all_packs.npz", allow_pickle=True)
reliable = trust["evidence_tier"] == "anchored_and_validated"     # 19% of parcels
```

For interactive exploration, see `notebooks/01_quickstart.ipynb` (Python) or the browser-only [HOMER Mapping Explorer](https://peach-lucien.github.io/homer/) (no install).

## What's in this repo

```
homer/
├── docs/                # 7-doc reading path + the published GUI (docs/index.html)
├── src/homer/           # The library — data, models, eval, viz, costs
├── pipeline/            # End-to-end reproduction scripts (02 → 08)
├── experiments/         # Anchor-pack experiments + ablations
├── notebooks/           # 4 interactive walkthroughs
├── tests/               # pytest
├── outputs/             # All generated artefacts (gitignored)
└── config/              # YAML configs for anchors
```

Documentation navigation hub: [`docs/README.md`](docs/README.md). To rebuild the explorer locally, run `python pipeline/08_build_gui.py --publish` — this regenerates `docs/index.html` from the current model.

## How the method works in one paragraph

We solve a **Fused Gromov-Wasserstein optimal transport** problem (POT's `entropic_semirelaxed_fused_gromov_wasserstein`) that finds a soft coupling π minimising (within-mouse FC + SC distance ↔ within-human FC + SC distance) + (cross-species xyz + anchor cost). The mouse row marginal is fixed uniform; the human column marginal is free (semirelaxed). The result is supervised by **21 Garin point anchors** (single-parcel) plus **7 region-anchor packs** (multi-parcel sub-region homologies from Bakken 2021 / May 2006 / Mori 2014 / Vogt 2019 / Janak & Tye 2015 / Strange 2014 / Wallis 2012). Anchors are *soft* by default — the FGW solver can violate the constraint if structural cost strongly disagrees. See [`docs/02_methods.md`](docs/02_methods.md).

## Honest limitations

- HOMER is supervised, not unsupervised. Held-out region CV recovers only 3.4 % top-1 — FC + SC alone don't encode reliable cross-species correspondences. The 37 % top-1 comes from anchor supervision, not structural recovery.
- The 100 % top-1 on pack-anchored regions is largely **by construction** — anchors match the validation sets.
- Per-parcel claims are region-level, not millimetre-level (mean argmax distance is 25-45 mm in well-anchored regions).
- Cerebellum and medulla are excluded from the parcellation.
- dlPFC homology is contested (opt-in in the lateral PFC pack).

See [`docs/05_limitations.md`](docs/05_limitations.md) for the full list.

## The four model levels

`homer.models` exposes four sklearn-style classes:

| Class | Use when |
|---|---|
| `UnsupervisedGW` | Sanity check (no anchors, no xyz) |
| `SupervisedFGW` | Anchors + xyz; no SC |
| **`MultimodalFGW`** | **Production: FC + SC + anchors + xyz** |
| `HierarchicalFGW` | Per-network sub-solves (best within-net FC) |

Plus two comparative additions kept as ablations: `FUGWModel` (unbalanced FGW) and a Knox 2019 voxel-SC variant. Neither moves the headline numbers.

## Citing

Manuscript in preparation. The repo bundles 21 Garin homologue anchors (Garin 2021) + 7 published anchor packs as listed in `docs/04_anchor_packs.md`. Beauchamp 2022 (eLife) provides external validation.

## License

MIT. See [`LICENSE`](LICENSE).
