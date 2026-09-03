# Frozen LEADS external confirmation

This test was specified before inspecting any translated LEADS result. The
external cohort is the multisite Longitudinal Early-Onset Alzheimer's Disease
Study (LEADS), released as public voxelwise maps in NeuroVault collection
[23001](https://neurovault.org/collections/23001/).

## Fixed hypothesis

The source subtypes and mouse targets are inherited unchanged from the La Joie
et al. discovery analysis:

- S1 / Typical -> amnestic-AD medial-temporal target (`ENTl`, `ENTm`, `CA1`,
  `CA3`, `DG`, `SUB`);
- S3 / Posterior -> PCA visual target (`VISp`, `VISl`, `VISal`, `VISam`,
  `VISpm`, `VISrl`, `VISa`, `VISpor`);
- S2 / Left Temporal -> lvPPA auditory/temporal-association target (`AUDp`,
  `AUDd`, `AUDv`, `TEa`).

No LEADS-derived structure was added to a target set.

## Fixed primary analysis and gate

The primary inputs are the authors' subtype-versus-rest baseline flortaucipir
T maps. Their voxelwise models adjust for SuStaIn stage, age, sex, education,
and Centiloid and are thresholded at uncorrected p<0.001.

The primary statistic is the discovery analysis's threshold-free joint
matched-target selectivity, using the canonical coupling and parcel-balanced
mouse target scores. The null uses 10,000 joint mirrored, bijective rotations
within human hemisphere. Confirmation requires:

1. positive selectivity for every subtype;
2. joint one-sided p<0.05 under canonical OTTER; and
3. joint one-sided p<0.05 after refitting OTTER without the directly relevant
   medial-temporal, visual, and auditory anchor packs.

Equal-structure target weighting is a sensitivity and is not part of the gate.

## Fixed unthresholded sensitivity

To test dependence on the authors' voxel threshold, the public unthresholded
mean-SUVR maps are matched on SuStaIn stage. Stage 0-8 is excluded because S2
has only six participants. Within bins 9-11, 12-14, and 15-20, each subtype is
contrasted with the equal mean of the other two; the three z-scored contrasts
are then averaged. The same mapping, statistic, null, and coupling control are
used.

## Interpretation boundary

This is an independent public multisite cohort and analysis release, but exact
participant non-overlap with the UCSF-led discovery sample cannot be proven
from aggregate maps alone because UCSF contributes to LEADS. The result should
therefore be described as a largely independent external confirmation unless
participant IDs or a centrally generated LEADS map excluding UCSF establish
complete independence.

The complete executable record is
`experiments/reverse_translation/10_ad_external_leads_confirmation.py` and its
machine-readable output is
`outputs/logs/reverse_translation_ad_leads_confirmation.json`.
