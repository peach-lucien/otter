# Overview

## What OTTER is

OTTER (**O**ptimal **T**ransport for **T**ranslation across **E**volutionary **R**elatives) is a Python package that learns probabilistic correspondences between mouse and human brain parcels. The output is a coupling matrix **π** of shape (1864 mouse parcels × 2094 human parcels) where `π[i, j]` is interpretable as "probability that mouse parcel *i* corresponds to human parcel *j*".

> **Note on sharpness and reconstruction.** The canonical coupling is fitted at ε = 0.05 and is deliberately soft: the median top-target probability is 0.31, above 0.5 for 20 % of parcels. Concentration is set by the regularisation rather than by anatomy, and re-fitting at ε = 0.005 gives a near-deterministic coupling with no gain in held-out recovery. Because the human marginal is free (semirelaxed FGW), the coupling can leave human parcels poorly reconstructed rather than forcing mass onto them. We report reconstruction accuracy (docs/03_results.md §5) rather than an uncovered-parcel percentage, which would depend on the threshold chosen. Each column of π is normalised before the push-forward, so the score reflects whether some mouse tissue is wired like the human parcel rather than how much mass that parcel received.

The method is **Fused Gromov-Wasserstein optimal transport** (POT's `entropic_semirelaxed_fused_gromov_wasserstein`), supervised by published mouse↔human anchor pairs.

## Who it's for

- **Cross-species researchers** translating mouse studies into human predictions (or vice versa) at the brain-region level
- **Method developers** wanting a calibrated baseline mouse↔human mapping to compare against
- **Reviewers** evaluating cross-species translational claims

## What you query

For any mouse parcel, OTTER returns a probability distribution over 2094 human parcels. Typical workflows:

```python
import numpy as np
from otter.data import load_cached

from otter.data import load_pi
pi = load_pi()                                   # pi_canonical.npy, 1864 × 2094

# Top-5 human partners for mouse parcel 1234
top5 = pi[1234].argsort()[::-1][:5]
```

For trust-tier filtering ("which mouse parcels can I rely on?"), load the multi-source trust map:

```python
trust = np.load("outputs/coupling/trust_multisource_canonical.npz", allow_pickle=True)
reliable = trust["evidence_tier"] == "anchored_and_validated"     # 31.5% of parcels
```

See `03_results.md` for what each tier means and how to read the headline numbers.

## What OTTER *is not*

- Not an unsupervised method. It requires anchor pairs (we ship 21 Garin point anchors + 26 region-anchor entries from 15 anchor packs).
- Not a voxel-level mapping. π is parcel-to-parcel. Mouse parcels span ~12-2837 voxels each.
- Not a cellular-resolution tool. It is spatial and connectivity-based; cell-type homology (BICCN, Allen Brain Cell Atlas) is a separate problem.
- Not validated for cerebellum or medulla (excluded from our parcellation).

## Headline number

For the **canonical π** (`pi_canonical.npy`, what `load_pi()` returns):

| Metric | Value |
|---|---:|
| Beauchamp 2022 region-level AUROC | **0.90** parcel-weighted, 0.93 unweighted (19/19 regions significant, FDR q < 0.05) |
| Beauchamp 2022 parcel-level top-1 | **57 %** |
| Mean centroid displacement from the expected homologue | **8.8 mm** (chance 25 mm) |
| Held-out recovery (41 units removed in turn, re-fit) | **AUROC 0.74** region-level; parcel-exact collapses to ~10 % |
| Memorisation control (curation overlapping each benchmark region removed) | 0.90 → **0.73**, 5/19 below chance |
| Median top-target probability | **0.31** (> 0.5 for 20 % of parcels) |
| Multi-source trust tier `anchored_and_validated` | **31.5 %** of parcels (validated tiers 55 %) |

See `03_results.md` for the full six-section results, the third-party validation table, and the caveats. See `04_anchor_packs.md` for the per-pack contribution.

## Project structure at a glance

```
otter/
├── src/otter/          # The library (data, models, eval, viz, costs)
├── pipeline/           # Reproduction scripts (02 → 07)
├── experiments/        # Anchor-pack runners + ablations
├── notebooks/          # 9 walkthroughs: quickstart, methodology, one per figure
├── docs/               # You are here
├── tests/              # pytest (~10s, 176 tests)
├── outputs/            # All generated artefacts (π, JSONs, figures)
└── config/             # YAML configs for anchors
```
