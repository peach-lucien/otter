# Overview

## What HOMER is

HOMER (**Hom**ology **E**stimation across species via **R**egional optimal transport) is a Python package that learns probabilistic correspondences between mouse and human brain parcels. The output is a coupling matrix **π** of shape (1864 mouse parcels × 2094 human parcels) where `π[i, j]` is interpretable as "probability that mouse parcel *i* corresponds to human parcel *j*".

> **Note on sharpness and coverage.** At the production regularisation (ε=0.005) the coupling is sharp/peaked, not broadly soft: ~67 % of mouse rows place essentially all mass on a single human parcel (median effective targets ≈ 1; an intended low-entropy regime, tunable via ε, raising ε softens it). Because the human marginal is free (semirelaxed FGW), the coupling maps the mouse atlas onto a subset of human parcels: ~53 % of human parcels receive negligible mass (< 1e-6; the figure is threshold-dependent, 41 % at machine zero), so reverse (human→mouse) queries have no source for about half of human parcels.

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
reliable = trust["evidence_tier"] == "anchored_and_validated"     # 31% of parcels
```

See `03_results.md` for what each tier means and how to read the headline numbers.

## What HOMER *is not*

- Not an unsupervised method. It requires anchor pairs (we ship 21 Garin point anchors + 26 region-anchor entries from 15 anchor packs).
- Not a voxel-level mapping. π is parcel-to-parcel. Mouse parcels span ~12-2837 voxels each.
- Not a cellular-resolution tool. It is spatial and connectivity-based; cell-type homology (BICCN, Allen Brain Cell Atlas) is a separate problem.
- Not validated for cerebellum or medulla (excluded from our parcellation).

## Headline number

For the **recommended π** (`pi_fc_plus_SC_with_all_packs.npy`):

| Metric | Value |
|---|---:|
| Beauchamp 2022 region-level AUROC | **0.85** (18/19 regions significant, FDR q < 0.05) |
| Beauchamp 2022 parcel-level top-1 | **45.7 %** (enrichment 50.6× over null) |
| Held-out recovery (41 units removed in turn, re-fit) | **AUROC 0.73** region-level; top-1 collapses to ~2 % |
| Region-level qualified top-3 (Beauchamp-22 set) | **100 %** |
| Bootstrap argmax stability (40 subject-resamples) | **98.2 %** |
| z-score vs permuted-anchor null | **+17.8** |
| Multi-source trust tier "anchored_and_validated" | 31 % of parcels |

See `03_results.md` for the full six-section results, the third-party validation table, and the caveats. See `04_anchor_packs.md` for the per-pack contribution.

## Project structure at a glance

```
homer/
├── src/homer/          # The library (data, models, eval, viz, costs)
├── pipeline/           # Reproduction scripts (02 → 07)
├── experiments/        # Anchor-pack runners + ablations
├── notebooks/          # 9 walkthroughs: quickstart, methodology, one per figure
├── docs/               # You are here
├── tests/              # pytest (~10s, 176 tests)
├── outputs/            # All generated artefacts (π, JSONs, figures)
└── config/             # YAML configs for anchors
```
