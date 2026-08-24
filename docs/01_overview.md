# Overview

OTTER (**O**ptimal **T**ransport for **T**ranslation across **E**volutionary
**R**elatives) estimates probabilistic correspondence between 1,864 mouse and
2,094 human brain parcels. The released output is a transport coupling π with
shape (1864, 2094).

An entry π[i, j] is a transport weight, not a calibrated probability of
homology. After normalising a row, it can be read as a distribution over human
targets for one mouse parcel:

    import numpy as np
    from otter.data import load_pi

    pi = load_pi()
    mouse_idx = 1234
    weights = pi[mouse_idx] / pi[mouse_idx].sum()
    top5 = np.argsort(weights)[::-1][:5]

The canonical coupling combines functional and structural connectivity, an
anchor-warped spatial cost, 21 Garin homology classes and 26 curated regional
correspondence entries. The mouse marginal is fixed and the human marginal is
free, so some human parcels can receive little mass.

## Appropriate uses

- Rank plausible human targets for a mouse parcel or region.
- Route a mouse spatial map to the human brain, or rank mouse structures for a
  human spatial map.
- Examine where human connectivity is reconstructed well or poorly from mouse
  connectivity.
- Refit the model under a documented ablation or supervision holdout.

## Interpretation

- OTTER is anatomically supervised; it is not unsupervised homology discovery.
- Region-level correspondence is more robust than parcel-exact assignment.
- The Beauchamp transcriptomic correspondences inform hyperparameter evaluation,
  and some benchmark territories overlap the anatomical scaffold. The stricter
  generalisation analysis removes overlapping supervision and refits the model
  target by target.
- Display labels in the explorer record anchor and benchmark-region membership
  together with internal stability summaries. They are interface metadata, not
  confidence estimates or additional validation results.
- Translated maps are spatial hypotheses rather than human measurements.

The parcellations exclude cerebellum and medulla. OTTER is parcel-level rather
than voxel- or cell-level, and it estimates correspondence between group-average
connectomes rather than individuals.

See [02_methods.md](02_methods.md) for the formulation,
[05_limitations.md](05_limitations.md) for interpretation limits and
[07_pipeline.md](07_pipeline.md) for the supported reproduction path.
