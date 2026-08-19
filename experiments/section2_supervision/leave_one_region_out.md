> The per-region table below comes from the fit without the anchor warp and is read for the
> ordering it shows rather than for its absolute values; the aggregate figures come from
> `outputs/logs/anchor_recovery_loo_combined_canonical.json`.

# Leave-one-region-out generalisation test (full production model)

For each Beauchamp pair, **all** curated supervision located in that region (its
Garin point anchor(s) *and* any region-anchor pack overlapping its mouse parcels)
is removed, the full FGW coupling is re-fit with everything else, and the region
is re-scored. This is the "would we recover this region if we had **not** curated
it" test for the production model (Garin + all 15 packs).

Script: `leave_one_region_out.py`. Raw data:
`otter/outputs/logs/beauchamp_leave_one_region_out.json`.
Harness reproduces the logged full-model per-pair top-1 exactly.

## Results (full → leave-one-out)

| region | top-1 | mass-in-region | centroid dist (mm) | random (mm) | n |
|---|---|---|---|---|---|
| Piriform | 1.00 → 0.00 | 1.00 → 0.00 | 7 → 10 | 71 | 52 |
| Subiculum | 1.00 → 0.00 | 0.99 → 0.00 | 13 → 45 | 63 | 35 |
| Field CA1 | 1.00 → 0.00 | 1.00 → 0.00 | 6 → 55 | 62 | 18 |
| Field CA3 | 1.00 → 0.00 | 1.00 → 0.00 | 11 → 32 | 62 | 25 |
| Dentate gyrus | 1.00 → 0.00 | 1.00 → 0.00 | 24 → 46 | 62 | 23 |
| Primary auditory | 1.00 → 0.00 | 1.00 → 0.00 | 15 → 48 | 58 | 9 |
| Primary motor | 1.00 → 0.00 | 1.00 → 0.00 | 12 → 23 | 66 | 53 |
| Inferior colliculus | 1.00 → 0.00 | 1.00 → 0.00 | 1 → 70 | 63 | 29 |
| Cortical subplate→amygdala | 0.97 → 0.02 | 0.97 → 0.02 | 5 → 29 | 69 | 58 |
| Superior colliculus | 0.96 → 0.00 | 0.96 → 0.00 | 1 → 60 | 59 | 55 |
| Thalamus | 0.33 → 0.31 | 0.33 → 0.31 | 11 → 11 | 58 | 103 |
| Caudoputamen | 0.32 → 0.10 | 0.32 → 0.10 | 10 → 14 | 63 | 143 |
| Primary somatosensory | 0.15 → 0.18 | 0.15 → 0.18 | 25 → 15 | 67 | 154 |
| Hypothalamus | 0.10 → 0.06 | 0.10 → 0.06 | 25 → 26 | 65 | 52 |
| Striatum ventral (NAcc)* | 0.08 → 0.08 | 0.08 → 0.08 | 6 → 6 | 67 | 26 |
| Pallidum | 0.06 → 0.02 | 0.06 → 0.02 | 16 → 17 | 61 | 51 |
| Anterior cingulate | 0.04 → 0.09 | 0.05 → 0.09 | 21 → 36 | 70 | 23 |
| Visual areas | 0.04 → 0.00 | 0.04 → 0.00 | 52 → 63 | 82 | 52 |
| Pons | 0.03 → 0.00 | 0.03 → 0.00 | 47 → 49 | 73 | 69 |

\* NAcc had no curated supervision to remove (its "held-out" equals its full
value); exclude it from the generalisation aggregate.

**Parcel-weighted aggregate:** top-1 0.46 → 0.08; mass 0.46 → 0.08;
chance centroid displacement is 25 mm on the canonical benchmark; the per-region distances in the table come from the fit without the anchor warp.

## Interpretation

1. **Parcel-exact homology requires curation.** Parcel-exact recovery collapses to
   roughly 10 % when a region's own supervision is held out, against 57 % for the
   full model. That full-model figure therefore largely checks that curated
   homologies are embedded, rather than demonstrating discovery.
2. **Connectivity provides coarse localisation on its own.** Held-out regions
   still route close to target: held-out region-level AUROC averages 0.74 across the 41 units, against 0.90 for the full model, and chance centroid displacement is 25 mm. Top-1 = 0 hid this; the
   distance metric reveals it (piriform 10 mm, motor 23 mm, thalamus 11 mm,
   striatum 6 mm, caudoputamen 14 mm, somatosensory 15 mm, pallidum 17 mm).
3. **Heterogeneous.** Some regions do not generalise even coarsely
   (inferior colliculus 70 mm, superior colliculus 60 mm, CA1 55 mm, visual
   63 mm); these are entirely anchor-dependent.

28 mm is coarse (~20% of brain span). The somatosensory pack lowers recovery
(0.15 with it, 0.18 without), consistent with its registry note.
