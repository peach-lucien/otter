# HOMER

**Hom**ology **E**stimation across species via **R**egional optimal transport.

A Python package that learns probabilistic cross-species correspondences between mouse and human brain parcels using **Fused Gromov–Wasserstein optimal transport**, anchored on published homologue pairs.

Output: a coupling matrix **π** of shape (1864 mouse parcels × 2094 human parcels) where `π[i, j]` is the model's estimated probability that mouse parcel *i* corresponds to human parcel *j*.

---

## Try it in your browser (no install)

**→ [Open the HOMER Mapping Explorer](https://peach-lucien.github.io/homer/)**

A self-contained 3D viewer for the production coupling: search a mouse region or parcel, see its top-K human partners ranked by coupling mass, toggle the cortical surface or mouse atlas shell, and inspect the trust evidence behind every prediction. No Python, no install, no backend. It is a single HTML file with the recommended model baked in. Use this if you want to *look at* HOMER. The rest of the README is for using it programmatically or reproducing it.

---

## What π is, in one sentence

**π is a connectional correspondence: connectional organisation transfers through it, and microstructure does not.**

That is the organising claim of the package, and it is what the validation shows. π is fitted on functional and structural connectivity. Things made of connectivity travel through it; things it never saw do not. The results below, including the places where HOMER fails, all follow from this.

## Headline numbers

The coupling is **sharp** (the top human partner carries a median probability of 1.0 and > 0.5 for **92 %** of mouse parcels), **homology-respecting** (mean self-mass on the 21-class homology diagonal **0.26**, versus 0.048 under a uniform mapping), and **topographically faithful** (distance between two mouse parcels predicts the distance between their routed human centroids at **r = 0.61**, against a permuted-coupling null of ≈ 0).

We scored the recommended coupling on Beauchamp 2022's external gene-expression benchmark, a transcriptomic homology set that never enters the fit. It reaches **region-level AUROC 0.85** and **45.7 %** parcel-level top-1, significant for 18 of 19 regions against a parcel-set permutation null (FDR q < 0.05).

### What carries the correspondence

We asked which ingredient of the model carries the correspondence. Removing the cost terms one at a time separates two quantities that move independently:

| cost terms | region-level (AUROC) | parcel-exact (top-1) | displacement |
|---|---:|---:|---:|
| connectivity only (GW on FC + SC) | 0.67 (chance) | 0.8 % | 35 mm |
| + spatial position | **0.87** (saturated) | 8 % | 28 mm |
| + curated anchors and packs | 0.85 (unchanged) | **46 %** | **17 mm** |

Connectivity and spatial position carry *which human region* a mouse region corresponds to. Curation carries *which parcel*. We then withheld the curation entirely, removing each of the 41 supervision units (15 Garin homology classes + 26 region packs) in turn and re-fitting the model. Region-level recovery holds at **AUROC 0.73** while parcel-exact recovery collapses to ~2 %.

Connectivity alone is unidentifiable rather than uninformative. Gromov–Wasserstein aligns two connectomes only up to relabelling, so with nothing to fix the global orientation the coupling cannot be placed. HOMER is therefore neither a connectivity-only method nor a landmark look-up. It needs both halves.

### What transfers through π

We next asked what survives the crossing. Three published cross-species results, none of them data HOMER saw, each tested under a **spatial-autocorrelation-preserving spin null**, a bar most cross-species analyses never set.

| test | modality | verdict |
|---|---|---|
| Mouse resting-state networks → human (Coletta 2020) | connectivity | **translates**: 6/10 top-match their homologue vs 1.2 expected, spin p = 0.002 |
| Principal FC gradient → human (Margulies / Huntenburg) | connectivity | **translates**: \|r\| = 0.54, spin p = 0.004 |
| Myelin (T1w:T2w) + cytoarchitecture (Fulcher 2019) | microstructure | **does not clear a spatial null**: spin p = 0.11 / 0.10 |

The routed microstructure maps do resemble the human myelin map (r = 0.37, 0.36) and they comfortably beat a permuted-π null. They do not beat spatial smoothness. We therefore do not claim that microstructure translates.

Conservation does reach *broad* cell classes: the excitatory − inhibitory contrast translates (r = 0.26, spin p = 0.001), while neuronal − glial, laminar and areal-type contrasts do not.

### Where π has no support

Semi-relaxed FGW frees the human marginal, so the coupling may leave human parcels uncovered, and it does, for **53 %** of them (mass < 1e-6), concentrated over association cortex. We asked what kind of absence this is. If it were molecular, a transcriptomic measure of mouse–human similarity should fall away from sensorimotor to association cortex alongside the connectional one. Only one of the two does:

- **connectivity coverage** collapses from sensorimotor to association cortex (+0.47 SD, spin p = 0.016)
- **transcriptomic similarity to mouse** does not (−0.16 SD, p = 0.45, n.s.)
- the **dissociation itself** is significant (+0.64 SD, spin p = 0.038)

The mouse has the parts; it does not have the wiring. HOMER localises, from connectivity alone, the conserved-but-rewired territory that reorganised in human cortical evolution.

That measurement then predicts disease. We correlated coverage, the mouse mass arriving at each human region, with the ENIGMA case-control effect size for cortical thinning across 30 Desikan–Killiany regions. Out of a 15-condition battery, only **bipolar disorder** (ρ = +0.64) and **schizophrenia** (ρ = +0.52) survive FDR: thinning is worst where the mouse cannot reach. Within those two disorders the relationship reverses in subcortex (ρ = −0.68 / −0.79; interaction p = 0.003 / 0.002). A mouse model cannot address their cortical signature, but it can address their subcortical one.

### Confidence grading

Every mouse parcel carries an evidence tier: **31 %** `anchored_and_validated`, **22 %** `validated_only`, **13 %** `anchored_only`, **14 %** `structural`, **21 %** `low_evidence`. The two validated tiers cover **52 %** of the brain.

The tier grades the resolution at which a prediction can be trusted rather than whether a homologue exists. Region-level recovery is essentially equal across the two validated tiers (AUROC 0.87 vs 0.88); parcel-exact recovery is not (top-1 0.69 vs 0.18). Trust cannot be read from the solver. At the production regularisation the coupling is sharply peaked everywhere, so the solver's own confidence is uncorrelated with accuracy. The grades are therefore external.

Full results in [`docs/03_results.md`](docs/03_results.md). One notebook per figure in [`notebooks/`](notebooks/).

## Install + quickstart (for programmatic use)

Skip this section if you only want to look at the couplings. Use the [explorer](https://peach-lucien.github.io/homer/) above. Install only if you need to query π in code, re-train the model, or extend it.

```bash
git clone <this-repo> homer && cd homer
conda env create -f env.yml && conda activate homer
pip install -e ".[dev]"
pytest -q                            # ~10s, runs on a bare checkout (synthetic fixtures)
python scripts/fetch_data.py         # pull the couplings + caches from Zenodo (~620 MB)
```

The repository ships code only. The coupling, the processed AnnData caches, and
the validation inputs live in a versioned Zenodo archive and are pulled by
`scripts/fetch_data.py`; the committed result logs in `outputs/logs/` already let
you read every headline number without downloading anything. See
[`DATA.md`](DATA.md) for the tiers and what each contains. Data DOI: [10.5281/zenodo.20746024](https://doi.org/10.5281/zenodo.20746024). If you call into the library before fetching, it prompts you to download then; set `HOMER_AUTO_FETCH=1` to fetch automatically whenever data is needed.

Query a region after installation (requires the fetched data):

```python
import numpy as np
from homer.data import load_cached, load_pi

M, _ = load_cached("mouse", cache_dir="outputs/anndata")
H, _ = load_cached("human", cache_dir="outputs/anndata")
pi = load_pi()    # recommended coupling (1864, 2094)

# Top-5 human partners for mouse parcel 1234
top5_idx = pi[1234].argsort()[::-1][:5]
top5_regions = H.var.iloc[top5_idx][["region", "x", "y", "z"]]
print(top5_regions)

# Filter by trust tier
trust = np.load("outputs/coupling/trust_multisource_all_packs.npz", allow_pickle=True)
reliable = trust["evidence_tier"] == "anchored_and_validated"     # 31% of parcels
```

`load_pi()` defaults to `pi_fc_plus_SC_with_all_packs.npy`, the recommended coupling. It does **not** default to `pi_fc_plus_SC.npy`, the base coupling, which is a different matrix and gives different answers.

For interactive exploration, see `notebooks/01_quickstart.ipynb` (Python) or the browser-only [HOMER Mapping Explorer](https://peach-lucien.github.io/homer/) (no install).

## Mouse data preprocessing

The mouse parcellation ships pre-warped voxel indices for every parcel. Allen
CCFv3 (`ns_center_ix` at 25 µm, `AS_ix` at 200 µm) and DSURQE, computed from a
nonlinear DSURQE → CCFv3 registration. The loader reads those indices directly
from the `.mat` file, so no separate mouse coordinate-alignment step is needed.

Entry points:

- `pipeline/00_external/01_mouse_sc.py`, builds `mouse_sc.npy`: the Allen Mouse
  Connectivity Atlas (Oh et al. 2014) projected onto the parcellation via each
  parcel's CCFv3 centre voxel.
- `pipeline/00_external/02_mouse_genes.py`, builds `mouse_genes.npy`: Allen ISH
  expression sampled over each parcel's CCFv3 voxel set.
- `src/homer/data/io.py`, exposes the per-parcel voxel-index fields for
  downstream code.

## What's in this repo

```
homer/
├── docs/                # 7-doc reading path + the published GUI (docs/index.html)
├── src/homer/           # The library (data, models, eval, viz, costs)
├── pipeline/            # End-to-end reproduction scripts (02 → 08)
├── experiments/         # Anchor-pack experiments + ablations
├── notebooks/           # 9 walkthroughs: quickstart, methodology, one per figure
├── tests/               # pytest
├── outputs/             # Result logs (committed) + generated artefacts (gitignored)
└── config/              # YAML configs for anchors
```

Documentation navigation hub: [`docs/README.md`](docs/README.md). To rebuild the explorer locally, run `python pipeline/08_build_gui.py --publish`. This regenerates `docs/index.html` from the current model.

## How the method works in one paragraph

We solve a **Fused Gromov-Wasserstein optimal transport** problem (POT's `entropic_semirelaxed_fused_gromov_wasserstein`) that finds a soft coupling π minimising (within-mouse FC + SC distance ↔ within-human FC + SC distance) + (cross-species xyz + anchor cost). The mouse row marginal is fixed uniform; the human column marginal is free (semirelaxed), so the coupling can report that a human parcel has **no** mouse counterpart. The result is supervised by **21 Garin point anchors** (single-parcel) plus **15 region-anchor packs** (26 multi-parcel sub-region homology entries curated from the published literature, see [`docs/04_anchor_packs.md`](docs/04_anchor_packs.md)). Anchors are *soft* by default. The FGW solver can violate the constraint if structural cost strongly disagrees. See [`docs/02_methods.md`](docs/02_methods.md).

## What HOMER does not do

- **It does not translate microstructure.** Routed mouse myelin and cytoarchitecture resemble the human myelin map (r = 0.37 / 0.36) but do not clear a spatial null (spin p = 0.11 / 0.10). π was fitted on connectivity and carries connectivity. Do not read a microstructural correspondence out of it. Fine molecular detail is weaker still: broad cell classes translate (excitatory − inhibitory, r = 0.26), laminar and areal-type contrasts do not.
- **It does not reach human association cortex.** 53 % of human parcels receive negligible mouse mass (mass < 1e-6), concentrated there. The coupling reports the absence rather than hiding it, and we treat it as a measurement, but it still means a mouse model cannot address phenotypes living in that territory. See the disease result above.
- **It is not a parcel-level oracle without supervision.** Curation buys parcel precision. Withhold it and region-level correspondence survives (AUROC 0.73) while parcel-exact recovery collapses (~2 % top-1). Trust the tier: parcel-level for `anchored_and_validated`, region-level across the validated tiers.
- **It does not localise better than a transcriptomic translator.** We once claimed it did; that was a reduction artefact. TransBrain's output is region-level, so scoring at parcel resolution flatters HOMER by construction. On region identity TransBrain leads on its own benchmark (AUROC 0.84 vs 0.79), and the difference is not significant (paired Wilcoxon p = 0.17). HOMER leads where the modality is connectional: the gradient (0.55 vs 0.42), round-trip fidelity (0.98/0.95/0.97 vs 0.89/0.82/0.83), sharpness (≈ 3 vs ≈ 60 effective target regions) and absence detection. They are complementary instruments rather than competitors.
- **Cerebellum and medulla** are excluded from the parcellation. **dlPFC homology** is contested and opt-in only.

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

## A note on numbers

Every statistic in this repo is written to a JSON in `outputs/logs/` by the script that computes it, and the notebooks **recompute each headline number and assert it against that log**. This is deliberate. A 2026 audit found statistics that had been typed into figure titles by hand and existed in no output file, a diffusion component selected by a hard-coded index that turned out to be the wrong axis, and two right numbers computed different ways sitting side by side in a table. `tools/check_manuscript_numbers.py` and `experiments/validation/00_validate_published_maps.py` exist to stop all three. If you add a result, write it to a log; do not type it into prose.

## Citing

Manuscript in preparation. The repo bundles 21 Garin homologue anchors (Garin 2021) + 15 published anchor packs (26 region-anchor entries, all in the recommended composition) as listed in `docs/04_anchor_packs.md`. Beauchamp 2022 (eLife) provides external validation.

## Acknowledgements

HOMER was built by the **S01 project** of the reTune CRC, a collaborative research centre between Würzburg and Berlin. The project is led by PIs **Robert Peach**, **Phillip Boehm-Sturm**, and **Martin Reich**, with postdoctoral researcher **Stefan Koch** and PhD student **Mario Perales**.

## License

MIT. See [`LICENSE`](LICENSE).
