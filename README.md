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

On Beauchamp 2022's external 22-pair gene-expression benchmark:

| Metric (recommended `pi_fc_plus_SC_with_all_packs.npy`) | Value |
|---|---:|
| Parcel-level top-1 | **39 %** (3.3× over the strict baseline) |
| Region-level qualified top-3 | **100 %** |
| Bootstrap argmax stability (40 subject resamples) | **97.8 %** |
| z-score vs permuted-anchor null | **+17.8** |

Independent third-party validation against **twelve cross-species papers** (Pagani 2026 autism subtypes, Margulies/Huntenburg gradient, Coletta 2020 RSN, BICCN cell types, ENIGMA cross-disorder, Whitesell 2021 DMN, Hodge 2019 layers, Pagani per-model, Fulcher 2019 multimodal gradient, Balsters 2020 frontal-cortex divergence, TransBrain 2025 head-to-head, Buckner & Krienen 2013 tethering) establishes a clean resolution boundary: HOMER preserves cross-species signal at the **regional / area / network level** but does not translate **broadly-distributed cortical class markers**, **within-area lamination**, or **disorder-specific signal**.

Multi-source trust map: 31 % of mouse parcels are `anchored_and_validated`, 13 % `anchored_only`, 24 % `validated_only`, 13 % `structural`, 20 % `low_evidence`.

Full result tables, per-paper snapshots, and the honest caveats live in [`docs/03_results.md`](docs/03_results.md). Showcase walkthroughs in [`notebooks/05-15`](notebooks/).

## Install + quickstart (for programmatic use)

Skip this section if you just want to look at the couplings — use the [explorer](https://peach-lucien.github.io/homer/) above. Install only if you need to query π in code, re-train the model, or extend it.

```bash
git clone <this-repo> homer && cd homer
conda env create -f env.yml && conda activate homer
pip install -e ".[dev]"
pytest -q                            # ~10s, 176 tests
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

## Mouse package: v2 is now the default

HOMER now consumes the **v2 mouse package** (`corrs_mouse_v2.mat`) by default. v2 ships pre-warped CCFv3 voxel indices for every parcel (`ns_center_ix` at 25 µm, `AS_ix` at 200 µm) derived from Paul's nonlinear DSURQE → CCFv3 elastix warp. The legacy v1 heuristic — a brute-force 48-permutation + centroid-translation transform built by `pipeline/00_external/00c_align_mouse_to_ccf.py` — is no longer needed for mouse data; the v2 loader reads CCFv3 voxel indices directly from the .mat file. The human side stays on v1.

**v2 entry points** (use these for the production build):

- `pipeline/00_external/01b_mouse_sc_v2.py` — rebuild `mouse_sc.npy` from `ns_center_ix`.
- `pipeline/00_external/02c_mouse_genes_v2.py` — rebuild `mouse_genes.npy` from `AS_ix`.
- `src/homer/io.py` v2 loader — exposes the new voxel-index fields for downstream code.

The redundant v1 mouse scripts (`01_mouse_sc.py`, `02_mouse_genes.py`, `02b_mouse_genes_direct.py`, `diagnose_allen_ish.py`) and the `warp_rebuild/` directory have been removed from `main` and preserved on the **`archive/v1-pipeline`** branch. `00c_align_mouse_to_ccf.py` and `_mouse_transform.py` are retained only because the `experiments/autism_subtypes/allen_expansion/` chain still imports the heuristic transform — migrating that experiment to the v2 `AS_ix` path will let those be retired too.

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

## Honest limitations

- HOMER is supervised, not unsupervised. Held-out region CV recovers only 3.4 % top-1 — FC + SC alone don't encode reliable cross-species correspondences. The 39 % top-1 comes from anchor supervision, not structural recovery.
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

Manuscript in preparation. The repo bundles 21 Garin homologue anchors (Garin 2021) + 15 published anchor packs (26 region-anchor entries, all in the recommended composition) as listed in `docs/04_anchor_packs.md`. Beauchamp 2022 (eLife) provides external validation.

## License

MIT. See [`LICENSE`](LICENSE).
