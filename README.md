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

π carries areal position on the cortical hierarchy, and with it the properties that vary across
that axis.

That is the organising claim, and it sets what follows. π is fitted on functional and structural
connectivity, but what travels through it is not limited to connectivity: microstructure comes
across too, at r = 0.47. What does not come across is anything varying through the cortical
depth. The boundary is areal against laminar, rather than connectional against everything else.

## Headline numbers

Every value here is recomputed in [`notebooks/`](notebooks/), one notebook per figure, and
checked against the manuscript.

The coupling concentrates on the homology diagonal (mean self-mass across the 21 Garin classes
0.40, against 0.048 under a size-matched uniform mapping) and preserves topography (distance
between two mouse parcels predicts distance between their routed human centroids at r = 0.53,
against a permuted-coupling null of ≈ 0). Each mouse parcel's best human partner carries a median
probability of 0.31, above 0.5 for 20 % of parcels, so the coupling spreads mass rather
than committing to one partner. That concentration is set by the entropic regularisation, not by
anatomy: re-fitting at ε = 0.005 gives a near-deterministic coupling with no gain in held-out
recovery, so sharpness is a dial rather than evidence of correctness.

Scored against Beauchamp 2022's transcriptomic homology set, which never enters the fit, the
coupling reaches region-level AUROC 0.90 (parcel-weighted across the 19 pairs; 0.93
unweighted) at 57 % parcel-level top-1, with mass enrichment significant for 19 of 19
regions under a parcel-set permutation null (FDR q < 0.05).

### What carries the correspondence

Removing the cost terms one at a time separates two quantities that move independently:

| cost terms | region-level (AUROC) | parcel-exact (top-1) | displacement |
|---|---:|---:|---:|
| connectivity only (GW on FC + SC) | 0.69 | 0 % | 29 mm |
| + anchor-warped spatial scaffold | **0.97** | 27 % | 11 mm |
| + curated anchors | 0.93 | 26 % | 11 mm |
| + region packs (production) | 0.90 | **57 %** | **9 mm** |

Connectivity and the spatial scaffold appear to carry which human region a mouse region maps to.
Curation carries which parcel. Withholding the curation entirely, by removing each of the 41
supervision units (15 Garin classes, 26 region packs) in turn and re-fitting, leaves region-level
recovery at held-out AUROC 0.74 while parcel-exact recovery collapses to roughly 10 %.

Worth noting that the spatial scaffold is itself fitted to the Garin landmark pairs, so the
ladder separates kinds of supervision rather than supervision from none. Connectivity alone is
unidentifiable rather than uninformative: Gromov–Wasserstein aligns two connectomes only up to
relabelling, so with nothing fixing the global orientation the coupling cannot be placed. HOMER
is therefore neither a connectivity-only method nor a landmark look-up.

### What transfers through π

Each test below uses data HOMER never saw and a spatial-autocorrelation-preserving spin null, a
bar most cross-species analyses do not set.

| test | modality | result |
|---|---|---|
| Mouse resting-state networks → human (Coletta 2020) | connectivity | 6/10 top-match their homologue against 1.0 expected, spin p = 0.002 |
| Principal FC gradient → human (Margulies / Huntenburg) | connectivity | \|r\| = 0.54, spin p = 0.032 |
| Myelin (T1w:T2w) → human myelin (Fulcher 2019) | microstructure | r = 0.47 over 388 of 400 Schaefer regions |
| Cytoarchitecture → human myelin (Fulcher 2019) | microstructure | r = 0.47 |

Grouping fourteen properties by their relation to the areal hierarchy, all nine tests in the
"hierarchy maps" and "varies along the hierarchy" groups clear their spin nulls, and none of the
five orthogonal to it does. The controlling comparison sits within one dataset: individual
cortical-layer marker genes, which retain areal variation, translate at mean r = 0.23 (6 of 7
significant), whereas layer contrasts built from the same genes, which remove the shared areal
component, do not (mean r = 0.07). Granular L4 − infragranular is the expected exception, since
cortical granularity is itself the areal hierarchy.

### Where π has no support

Semi-relaxed FGW frees the human marginal, so the coupling may leave human parcels poorly
reconstructed. Reconstruction-coverage asks how well each human parcel's connectivity fingerprint
is rebuilt by routing mouse connectivity through π. It runs high over sensorimotor, auditory and
visual territory and low over prefrontal and lateral temporal cortex.

That deficit tracks cortical expansion. Six of seven published maps clear a spin null: coverage
falls with macaque→human expansion (ρ = −0.47, spin p = 0.003), mouse→human expansion (−0.32,
p = 0.001), the sensorimotor–association axis (−0.33, p = 0.017) and the principal gradient
(−0.28, p = 0.017). Only the T1w:T2w myelin map does not (+0.24, n.s.).

The deficit is network-shaped rather than a diffuse falloff. Control B, covering dorsolateral and
rostrolateral prefrontal cortex, is the only network significantly below the cortical mean
(−0.69 SD, spin p = 0.006), while transcriptomic similarity to mouse over the same parcels is
flat (−0.18 SD, p = 0.39). The mouse appears to have the parts without the wiring.

Coverage does not, however, resolve disorders. Correlating it with case-control cortical-thickness
effect sizes is null for all seven ENIGMA maps tested (minimum spin p = 0.15), and weighting each
disorder's thinning burden by reachability is null for all six disorders (minimum p = 0.13). The
test can detect a hierarchy-aligned effect when one is present: run identically, the myelin map
flags bipolar disorder (p = 0.028) and major depression (p = 0.011) in the same data.

### Confidence grading

Every mouse parcel carries an evidence tier: 31.5 % `anchored_and_validated`, 23.8 %
`validated_only`, 12.2 % `anchored_only`, 10.9 % `structural`, 21.6 % `low_evidence`.
The two validated tiers cover 55 % of the brain.

The tier grades the resolution at which a prediction can be trusted rather than whether a
homologue exists. Parcel-exact recovery separates the two validated tiers (top-1 0.70 against
0.39). Trust cannot be read from the solver: across parcels without anchor supervision, the
coupling's own concentration predicts top-1 accuracy at r = 0.06 and bootstrap stability at
r = −0.04, neither significant. Because the regularisation sets concentration directly, a
confident-looking coupling can be produced on demand, so the grades are external by necessity.

Full results in [`docs/03_results.md`](docs/03_results.md). One notebook per figure in
[`notebooks/`](notebooks/).

## Install + quickstart (for programmatic use)

Skip this section if you only want to look at the couplings. Use the [explorer](https://peach-lucien.github.io/homer/) above. Install only if you need to query π in code, re-train the model, or extend it.

```bash
git clone <this-repo> homer && cd homer
conda env create -f env.yml && conda activate homer
pip install -e ".[dev]"
pytest -q                            # ~10s, runs on a bare checkout (synthetic fixtures)
python scripts/fetch_data.py         # pull the couplings + caches from Zenodo (~735 MB)
```

The repository ships code only. The coupling, the processed AnnData caches, and
the validation inputs live in a versioned Zenodo archive and are pulled by
`scripts/fetch_data.py`; the committed result logs in `outputs/logs/` already let
you read every headline number without downloading anything. See
[`DATA.md`](DATA.md) for the tiers and what each contains. Data DOI: [10.5281/zenodo.21458106](https://doi.org/10.5281/zenodo.21458106). If you call into the library before fetching, it prompts you to download then; set `HOMER_AUTO_FETCH=1` to fetch automatically whenever data is needed.

Query a region after installation (requires the fetched data):

```python
import numpy as np
from homer.data import load_cached, load_pi

M, _ = load_cached("mouse", cache_dir="outputs/anndata")
H, _ = load_cached("human", cache_dir="outputs/anndata")
pi = load_pi()    # canonical coupling (1864, 2094)

# Top-5 human partners for mouse parcel 1234
top5_idx = pi[1234].argsort()[::-1][:5]
top5_regions = H.var.iloc[top5_idx][["region", "x", "y", "z"]]
print(top5_regions)

# Filter by trust tier
trust = np.load("outputs/coupling/trust_multisource_canonical.npz", allow_pickle=True)
reliable = trust["evidence_tier"] == "anchored_and_validated"     # 31% of parcels
```

`load_pi()` defaults to `pi_canonical.npy`, the canonical coupling used throughout the paper. Always call `load_pi()` rather than loading a filename: the earlier couplings (`pi_fc_plus_SC*.npy`) are retired, give different answers, and are kept only so that published comparisons remain reproducible. `pi_provenance()` returns the file and its sha256.

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
├── notebooks/           # 8 walkthroughs: quickstart, methodology, one per figure
├── tests/               # pytest
├── outputs/             # Result logs (committed) + generated artefacts (gitignored)
└── config/              # YAML configs for anchors
```

Documentation navigation hub: [`docs/README.md`](docs/README.md). To rebuild the explorer locally, run `python pipeline/08_build_gui.py --publish`. This regenerates `docs/index.html` from the current model.

## How the method works in one paragraph

We solve a **Fused Gromov-Wasserstein optimal transport** problem (POT's `entropic_semirelaxed_fused_gromov_wasserstein`) that finds a soft coupling π minimising (within-mouse FC + SC distance ↔ within-human FC + SC distance) + (cross-species xyz + anchor cost). The mouse row marginal is fixed uniform; the human column marginal is free (semirelaxed), so the coupling can report that a human parcel has **no** mouse counterpart. The result is supervised by **21 Garin point anchors** (single-parcel) plus **15 region-anchor packs** (26 multi-parcel sub-region homology entries curated from the published literature, see [`docs/04_anchor_packs.md`](docs/04_anchor_packs.md)). Anchors are *soft* by default. The FGW solver can violate the constraint if structural cost strongly disagrees. See [`docs/02_methods.md`](docs/02_methods.md).

## What HOMER does not do

- **It does not translate microstructure.** Routed mouse myelin and cytoarchitecture resemble the human myelin map (r = 0.37 / 0.36) but do not clear a spatial null (spin p = 0.11 / 0.10). π was fitted on connectivity and carries connectivity. Do not read a microstructural correspondence out of it. Fine molecular detail is weaker still: broad cell classes translate (excitatory − inhibitory, r = 0.26), laminar and areal-type contrasts do not.
- **It reconstructs association cortex poorly.** Reconstruction-coverage runs low over prefrontal and lateral temporal cortex, with dorsolateral prefrontal cortex the clearest case (Control B, −0.69 SD below the cortical mean). The coupling reports the shortfall rather than hiding it, and we read it as a measurement, but a mouse model still cannot address phenotypes living in that territory. Note that uncovered-parcel percentages quoted in earlier drafts were threshold-dependent and have been dropped.
- **It is not a parcel-level oracle without supervision.** Curation buys parcel precision. Withhold it and region-level correspondence largely survives (held-out mean AUROC 0.74) while parcel-exact recovery collapses to roughly 10 % top-1. Trust the tier: parcel-level for `anchored_and_validated`, region-level across the validated tiers.
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

Every statistic in this repo is written to a JSON in `outputs/logs/` by the script that computes it, and the notebooks recompute each headline number and check it against the value printed in the manuscript. This is deliberate. A 2026 audit found statistics that had been typed into figure titles by hand and existed in no output file, a diffusion component selected by a hard-coded index that turned out to be the wrong axis, and two right numbers computed different ways sitting side by side in a table. `tools/audit_pi.py` and `experiments/validation/00_validate_published_maps.py` exist to stop all three. If you add a result, write it to a log; do not type it into prose.

## Citing

Manuscript in preparation. The repo bundles 21 Garin homologue anchors (Garin 2021) + 15 published anchor packs (26 region-anchor entries, all in the recommended composition) as listed in `docs/04_anchor_packs.md`. Beauchamp 2022 (eLife) provides external validation.

## Acknowledgements

HOMER was built by the **S01 project** of the reTune CRC, a collaborative research centre between Würzburg and Berlin. The project is led by PIs **Robert Peach**, **Phillip Boehm-Sturm**, and **Martin Reich**, with postdoctoral researcher **Stefan Koch** and PhD student **Mario Perales**.

## License

MIT. See [`LICENSE`](LICENSE).
