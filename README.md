# HOMER

**Hom**ology **E**stimation across species via **R**egional optimal transport.

A Python package that learns probabilistic cross-species correspondences between mouse and human brain parcels using **Fused Gromov–Wasserstein optimal transport**, anchored on published homologue pairs.

Output: a coupling matrix **π** of shape (1864 mouse parcels × 2094 human parcels) where `π[i, j]` is the model's estimated probability that mouse parcel *i* corresponds to human parcel *j*.

---

## Try it in your browser (no install)

**→ [Open the HOMER Mapping Explorer](https://peach-lucien.github.io/homer/)**

A self-contained 3D viewer for the production coupling: search a mouse region or parcel, see its top-K human partners ranked by coupling mass, toggle the cortical surface or mouse atlas shell, and inspect the trust evidence behind every prediction. No Python, no install, no backend — it's a single HTML file with the recommended model baked in. Use this if you want to *look at* HOMER. The rest of the README is for using it programmatically or reproducing it.

---

## Headline numbers

On Beauchamp 2022's external 22-pair gene-expression benchmark:

| Metric (recommended `pi_fc_plus_SC_with_all_packs.npy`) | Value |
|---|---:|
| Parcel-level top-1 (weighted by n parcels) | **45.7 %** (50.6× over the permuted-anchor null; 0.9 % chance) |
| Region-level qualified top-3 | **100 %** |
| Bootstrap argmax stability (40 subject resamples) | **98.2 %** |
| z-score vs permuted-anchor null | **+17.8** |

> **Read these honestly:** they are **anchor-supervised** numbers — the recommended π is supervised on published homologues, and the high top-1 is partly *by construction* (anchors overlap the validation set). Held-out region cross-validation (drop a region's anchor, then predict it) recovers **3.4 % top-1 (~7× chance)**, and zeroing the spatial (xyz) prior collapses top-1 to chance. FC+SC alone don't encode reliable correspondence — the *supervision* carries it. HOMER's value is a calibrated, anchor-informed coupling, not unsupervised homology discovery.

Independent third-party validation against **twelve cross-species papers**, evaluated against **spatial-autocorrelation-preserving spin nulls** (a bar most cross-species analyses never set). HOMER's **discrete network/homology correspondences are specific and survive these nulls**: Coletta 2020 resting-state networks (6/10 diagonal-argmax, spin-null **p = 0.002**) and the Pagani 2026 network bridge (4/8, **p = 0.026**). **Specific structural maps survive too**: the Fulcher 2019 mouse myelin proxy and cytoarchitecture both reproduce the human myelin map (spin **p = 0.021 / 0.010**). It also passes two negative-control / falsification tests (Balsters 2020 frontal-cortex divergence, Buckner & Krienen 2013 tethering) and a head-to-head benchmark against the TransBrain 2025 sibling method. The one place it does not beat the null is the **most generic continuous fields** — the Margulies/Huntenburg principal gradient and the Pagani subtype Δ-matrix — which are *matched* but do **not** exceed spatial-autocorrelation expectation under a spin test (p ≈ 0.16–0.22); we report this rather than over-claim it. So HOMER's reliable scope is **categorical region/network correspondence and specific structural maps**, not the broadest smooth gradients.

Multi-source trust map: 31 % of mouse parcels are `anchored_and_validated`, 13 % `anchored_only`, 24 % `validated_only`, 13 % `structural`, 20 % `low_evidence`.

Full result tables, per-paper snapshots, and the honest caveats live in [`docs/03_results.md`](docs/03_results.md). Showcase walkthroughs in [`notebooks/05-15`](notebooks/).

## Install + quickstart (for programmatic use)

Skip this section if you just want to look at the couplings — use the [explorer](https://peach-lucien.github.io/homer/) above. Install only if you need to query π in code, re-train the model, or extend it.

```bash
git clone <this-repo> homer && cd homer
conda env create -f env.yml && conda activate homer
pip install -e ".[dev]"
pytest -q                            # ~10s, runs on a bare checkout (synthetic fixtures)
python scripts/fetch_data.py         # pull the coupling + caches from Zenodo (~173 MB)
```

The repository ships code only. The coupling, the processed AnnData caches, and
the validation inputs live in a versioned Zenodo archive and are pulled by
`scripts/fetch_data.py`; the committed result logs in `outputs/logs/` already let
you read every headline number without downloading anything. See
[`DATA.md`](DATA.md) for the tiers and what each contains. Data DOI: [10.5281/zenodo.20733163](https://doi.org/10.5281/zenodo.20733163).

Query a region after installation (requires the fetched data):

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

## Mouse data preprocessing

The mouse parcellation ships pre-warped voxel indices for every parcel — Allen
CCFv3 (`ns_center_ix` at 25 µm, `AS_ix` at 200 µm) and DSURQE — computed from a
nonlinear DSURQE → CCFv3 registration. The loader reads those indices directly
from the `.mat` file, so no separate mouse coordinate-alignment step is needed.

Entry points:

- `pipeline/00_external/01_mouse_sc.py` — builds `mouse_sc.npy`: the Allen Mouse
  Connectivity Atlas (Oh et al. 2014) projected onto the parcellation via each
  parcel's CCFv3 centre voxel.
- `pipeline/00_external/02_mouse_genes.py` — builds `mouse_genes.npy`: Allen ISH
  expression sampled over each parcel's CCFv3 voxel set.
- `src/homer/data/io.py` — exposes the per-parcel voxel-index fields for
  downstream code.

## What's in this repo

```
homer/
├── docs/                # 7-doc reading path + the published GUI (docs/index.html)
├── src/homer/           # The library — data, models, eval, viz, costs
├── pipeline/            # End-to-end reproduction scripts (02 → 08)
├── experiments/         # Anchor-pack experiments + ablations
├── notebooks/           # 15 interactive walkthroughs (4 core + 11 third-party showcases)
├── tests/               # pytest
├── outputs/             # All generated artefacts (gitignored)
└── config/              # YAML configs for anchors
```

Documentation navigation hub: [`docs/README.md`](docs/README.md). To rebuild the explorer locally, run `python pipeline/08_build_gui.py --publish` — this regenerates `docs/index.html` from the current model.

## How the method works in one paragraph

We solve a **Fused Gromov-Wasserstein optimal transport** problem (POT's `entropic_semirelaxed_fused_gromov_wasserstein`) that finds a soft coupling π minimising (within-mouse FC + SC distance ↔ within-human FC + SC distance) + (cross-species xyz + anchor cost). The mouse row marginal is fixed uniform; the human column marginal is free (semirelaxed). The result is supervised by **21 Garin point anchors** (single-parcel) plus **15 region-anchor packs** (26 multi-parcel sub-region homology entries curated from the published literature — see [`docs/04_anchor_packs.md`](docs/04_anchor_packs.md)). Anchors are *soft* by default — the FGW solver can violate the constraint if structural cost strongly disagrees. See [`docs/02_methods.md`](docs/02_methods.md).

## What HOMER does not do

HOMER captures cross-species correspondence at the **categorical region / area / network** level. It is **not** trustworthy — or is untested — for:

- **Smooth continuous maps (gradients, per-network Δ-matrices).** These are *matched* but do **not** exceed spatial-autocorrelation expectation under a spin test (Margulies principal gradient |r|=0.40, spin p≈0.16–0.22; Pagani subtype Δ-matrix r=0.55, spin p=0.19). Smooth maps are the hardest cross-species target — easy to match by smoothness, hard to prove specific. Use HOMER for *which network/region corresponds*, not for translating a continuous field.
- **Fine molecular detail is weaker than region-level signal.** Cell-type markers (BICCN, 13/23 significant) and cortical layer markers (Hodge, 6/7 significant) translate, but only moderately — r ≈ 0.1–0.2. Treat gene-spatial predictions as suggestive, not quantitative.
- **Disorder-specific signal** — predictions for autism vs schizophrenia vs ADHD correlate at r > 0.97; HOMER captures a generic psychiatric perturbation geometry, not disorder-specific biology.
- **Millimetre-level parcel claims** — per-parcel argmax distances are 25–45 mm even in well-anchored regions; treat results as region-level.
- **Unsupervised recovery** — held-out region CV recovers only 3.4 % top-1, so FC + SC alone don't encode reliable correspondences. The headline numbers come from anchor supervision, and the 100 % pack-anchored top-1 is largely **by construction** (anchors match the validation sets).
- **Cerebellum and medulla** — excluded from the parcellation. **dlPFC homology** is contested and opt-in only.

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

Manuscript in preparation. The repo bundles 21 Garin homologue anchors (Garin 2021) + 15 published anchor packs (26 region-anchor entries, all in the recommended composition) as listed in `docs/04_anchor_packs.md`. Beauchamp 2022 (eLife) provides external validation.

## Acknowledgements

HOMER was built by the **S01 project** of the reTune CRC, a collaborative research centre between Würzburg and Berlin. The project is led by PIs **Robert Peach**, **Phillip Boehm-Sturm**, and **Martin Reich**, with postdoctoral researcher **Stefan Koch** and PhD student **Mario Perales**.

## License

MIT. See [`LICENSE`](LICENSE).
