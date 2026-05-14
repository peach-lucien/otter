# Overview

## What HOMER is

HOMER (**Hom**ology **E**stimation across species via **R**egional optimal transport) is a Python package that learns soft probabilistic correspondences between mouse and human brain parcels. The output is a coupling matrix **π** of shape (1864 mouse parcels × 2094 human parcels) where `π[i, j]` is interpretable as "probability that mouse parcel *i* corresponds to human parcel *j*".

The method is **Fused Gromov-Wasserstein optimal transport** (POT's `entropic_semirelaxed_fused_gromov_wasserstein`), supervised by published mouse↔human anchor pairs.

## Who it's for

- **Cross-species researchers** translating mouse studies into human predictions (or vice versa) at the brain-region level
- **Method developers** wanting a calibrated baseline mouse↔human mapping to compare against
- **Reviewers** evaluating cross-species translational claims

## What you query

For any mouse parcel, HOMER returns a probability distribution over 2094 human parcels. Typical workflows:

```python
import numpy as np
from homer.data import load_cached

pi = np.load("outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")    # 1864 × 2094

# Top-5 human partners for mouse parcel 1234
top5 = pi[1234].argsort()[::-1][:5]
```

For trust-tier filtering ("which mouse parcels can I rely on?"), load the multi-source trust map:

```python
trust = np.load("outputs/coupling/trust_multisource_all_packs.npz", allow_pickle=True)
reliable = trust["evidence_tier"] == "anchored_and_validated"     # 19% of parcels
```

See `03_results.md` for what each tier means and how to read the headline numbers.

## What HOMER *is not*

- Not an unsupervised method — it requires anchor pairs (we ship 21 Garin point anchors + 7 region-anchor packs).
- Not a voxel-level mapping — π is parcel-to-parcel. Mouse parcels span ~12-2837 voxels each.
- Not a cellular-resolution tool — it's spatial and connectivity-based; cell-type homology (BICCN, Allen Brain Cell Atlas) is a separate problem.
- Not validated for cerebellum or medulla (excluded from our parcellation).

## Headline number

For the **recommended π** (`pi_fc_plus_SC_with_all_packs.npy`, fits with 5 default anchor packs):

| Metric | Value |
|---|---:|
| Beauchamp 2022 anchor-overlapping top-1 | **37 %** (vs 12 % for production point-anchor only) |
| Beauchamp top-5 | **46 %** |
| Mean rank of correct human partner / 2094 | **85** (in the top 4 % on average) |
| Region-level qualified top-3 (Beauchamp-22 set) | **100 %** |
| Bootstrap argmax stability (40 subject-resamples) | **97.8 %** |
| Multi-source trust tier "anchored_and_validated" | 19 % of parcels |

See `03_results.md` for the full numbers, breakdowns, and honest caveats. See `04_anchor_packs.md` for the per-pack contribution.

## Project structure at a glance

```
homer/
├── src/homer/          # The library (data, models, eval, viz, costs)
├── pipeline/           # Reproduction scripts (02 → 07)
├── experiments/        # Anchor-pack runners + ablations
├── notebooks/          # 4 interactive walkthroughs
├── docs/               # You are here
├── tests/              # pytest (~10s, 161 tests)
├── outputs/            # All generated artefacts (π, JSONs, figures)
└── config/             # YAML configs for anchors
```
