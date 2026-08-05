# Results

Every number here is recomputed in `notebooks/`, one notebook per figure, and checked against the
value printed in the manuscript. Where a result requires re-fitting the model rather than scoring
it, the notebook says so and names the producing script. If a notebook does not reproduce a number
on this page, the notebook is right and this page is a bug.

---

## The organising claim

π carries areal position on the cortical hierarchy, and with it the properties that vary across
that axis.

π is fitted on functional and structural connectivity, but what travels through it is not limited
to connectivity. Microstructure comes across, at |r| = 0.50 and 0.53 over parcels and r = 0.47
against the human myelin map over regions. What does not come across is anything varying
through the cortical depth. The boundary is areal against laminar, rather than connectional
against everything else.

## Synthesis

1. The coupling is calibrated, and deliberately soft. Median top-target probability 0.31, above
   0.5 for 20 % of parcels. Concentration is set by the entropic regularisation, not by anatomy.
2. Connectivity and an anchor-warped spatial scaffold appear to set which human *region* a mouse
   region maps to. Curated anchors and packs set which *parcel*.
3. Translation follows the areal hierarchy. Networks, the principal gradient, myelin,
   cytoarchitecture and hierarchy-aligned cell densities all clear spin nulls; laminar contrasts
   and spatially uniform cell classes do not.
4. Against a state-of-the-art transcriptomic translator the two are level on region-level
   accuracy, on that method's own benchmark, and OTTER leads on the other six capability axes.
5. Where the mouse cannot rebuild human connectivity is a network-shaped territory tracking
   cortical expansion, with dorsolateral prefrontal cortex the clearest case.
6. The coupling turns a mouse experiment into a falsifiable human prediction. It does not resolve
   which disorder a mouse can model.

## The coupling files

`load_pi()` returns `pi_canonical.npy`, the coupling used throughout the paper. It combines region
packs with an anchor-warped spatial cost at ε = 0.05 and xyz weight 0.25, both selected by nested
cross-validation on held-out Beauchamp homologies.

| file | what it is | use |
|---|---|---|
| `outputs/coupling/pi_canonical.npy` | canonical coupling | what `load_pi()` returns |
| `outputs/coupling/pi_canonical_sharp.npy` | same recipe at ε = 0.005 | showcase variant; sharper, no more accurate |
| `outputs/coupling/pi_fc_plus_SC*.npy` | pre-warp couplings | retired; kept to reproduce published comparisons |

Verify by hash rather than by filename. `pi_provenance()` returns the file and its sha256, and
`tools/audit_pi.py` checks that every live analysis is on the canonical one.

## 1 · The coupling is calibrated

Mass concentrates on the homology diagonal. Mean self-mass across the 21 Garin classes is 0.40,
against 0.048 under a size-matched uniform mapping. Routing preserves topography, with the
distance between two mouse parcels predicting the distance between their routed human centroids at
r = 0.53 against a permuted-coupling null of ≈ 0.

The coupling is soft. Each mouse parcel's best human partner carries a median probability of 0.31,
above 0.5 for 20 % of parcels. Re-fitting at ε = 0.005 gives a near-deterministic coupling (median
0.96, above 0.5 for 90 %) with no gain in held-out homology recovery, so sharpness is a dial
rather than evidence of correctness. We select ε by held-out recovery and leave the spread visible.

### Evidence tiers

Each mouse parcel is graded on two external lines of evidence, neither of which is the solver's own
confidence: membership in a curated anchor, and independent reproduction of a published homology.

| tier | share of mouse parcels |
|---|---:|
| `anchored_and_validated` | 31.5 % |
| `validated_only` | 23.8 % |
| `anchored_only` | 12.2 % |
| `structural` | 10.9 % |
| `low_evidence` | 21.6 % |

The two validated tiers cover 55 % of the brain. The tier grades the resolution at which a
prediction can be trusted rather than whether a homologue exists. Parcel-exact recovery separates
the two validated tiers, at top-1 0.70 against 0.39.

Trust cannot be read from the solver. Across parcels without anchor supervision, the coupling's own
concentration predicts top-1 accuracy at r = 0.06 and bootstrap stability at r = −0.04, neither
significant. Since the regularisation sets concentration directly, a confident-looking coupling can
be produced on demand, which is why the grades are external.

## 2 · What carries cross-species homology

Scored against Beauchamp 2022's transcriptomic homology set, which never enters the fit, the
coupling reaches region-level AUROC 0.90 parcel-weighted across the 19 pairs (0.93 unweighted) at
57 % parcel-level top-1, with mass enrichment significant for 19 of 19 regions under a parcel-set
permutation null (FDR q < 0.05). Mean centroid displacement from the expected homologue is 8.83 mm (parcel-weighted, the value the manuscript quotes),
against a chance displacement of 25 mm.

Aggregates are weighted by parcel count. Regions differ roughly fifty-fold in size, so an
unweighted mean lets a 5-parcel region count as much as a 250-parcel one; both are reported because
the difference is easy to trip over.

Removing the cost terms one at a time, re-fitting at each stage:

| cost terms | AUROC | top-1 | mass-in-region | displacement |
|---|---:|---:|---:|---:|
| connectivity only (GW on FC + SC) | 0.69 | 0 % | 0.01 | 29 mm |
| + anchor-warped spatial scaffold | 0.97 | 27 % | 0.23 | 11 mm |
| + curated anchors | 0.93 | 26 % | 0.22 | 11 mm |
| + region packs (production) | 0.90 | 57 % | 0.54 | 9 mm |

Region-level recovery rises once the spatial scaffold is added and does not improve with curation;
parcel-exact recovery improves only with the anchors and packs. The spatial scaffold is itself
fitted to the Garin landmark pairs, so the ladder separates kinds of supervision rather than
supervision from none.

Connectivity alone is unidentifiable rather than uninformative. Gromov–Wasserstein aligns two
connectomes only up to relabelling, so with nothing fixing the global orientation the coupling
cannot be placed.

### Withholding the curation

Removing each of the 41 combined supervision units (15 Garin classes, 26 region packs) in turn and
re-fitting leaves held-out AUROC at a mean of 0.74, with 7 of 41 units below chance, predominantly
hippocampal subfields and fine somatotopic subdivisions. Parcel-exact recovery collapses to roughly
10 %.

### The memorisation control

For each of the 19 Beauchamp regions, removing the curation overlapping it and re-fitting leaves
parcel-weighted AUROC at 0.90 → 0.73, with 5 of 19 below chance. Agreement with the benchmark is
therefore not memorised curation, though the bound on parcel-exact recovery is real.

## 3 · What transfers through π

Each test uses data OTTER never saw and a spin null preserving spatial autocorrelation.

### Networks

Routing the mouse resting-state networks of Coletta et al. through π assigns 6 of 10 to their
like-named human network, against a spin-null expectation of 1.0 (p = 0.002). Nine of eleven
networks map to a more compact human territory than the null.

### The gradient

Routing the mouse principal functional-connectivity gradient predicts the observed human gradient
at |r| = 0.54 across all 2,094 parcels and |r| = 0.56 at region level, exceeding both a permuted-π
null (|r| = 0.07, p < 0.001) and a spin null (p = 0.032). The correspondence survives reduction to
discrete structure. The rank order of the nine human networks along the gradient is recovered at
ρ = 0.80 (spin p = 0.029), and a three-tier discretisation is classified at 56 % against 33 %
chance (p = 0.001).

### Microstructure

Two independent mouse measurements routed through π both predict the human HCP myelin map, the
T1w:T2w proxy and cytoarchitectural type. At parcel level each clears a translation null, which
rotates the mouse input and routes it through π unchanged, at |r| = 0.50 over 1,789 parcels
(p = 0.005) and |r| = 0.53 over 1,787 parcels (p = 0.003). Aggregated to the 388 of 400 Schaefer
regions the coupling reaches, each correlates with the human myelin map at r = 0.47.

One number stood for two measurements because the region-level values round together, 0.470 for
the myelin proxy and 0.473 for cytoarchitecture. The parcel-level values separate them.

Both mouse inputs are coarse. The myelin proxy takes 39 distinct values across the mouse cortex
and cytoarchitectural type takes five, so the nulls account for their resolution but the parcel
count is not a degrees-of-freedom count.

> Reversed in July 2026. An earlier draft reported r = 0.37 / 0.36 failing a spin null at
> p = 0.11 / 0.10 and concluded that microstructure does not translate. Those values came from the
> retired pre-warp coupling, and the conclusion also depended on a bug in `principal_gradient()`
> that returned an anterior-posterior spatial axis rather than the hierarchy. On the canonical
> coupling both measurements clear their nulls. The earlier claim is withdrawn.

### The pattern

Grouping fourteen properties by their relation to the areal hierarchy, all nine tests in the
"hierarchy maps" and "varies along the hierarchy" groups clear their spin nulls, and none of the
five orthogonal to it does.

The comparison is internally controlled. Eight of the fourteen measures are cell-class maps scored
on the same 2,094 parcels against the same null, and they span the full range, from −0.03 for
microglial density to +0.35 for the neuronal-glial contrast. What separates them is their relation
to the areal hierarchy rather than how they were measured. Granular L4 − infragranular is the
expected exception among the laminar contrasts at r = 0.19, since cortical granularity is itself
areal.

> Withdrawn July 2026. An earlier version of this page offered individual cortical-layer marker
> genes as the control, at mean r = 0.23 with 6 of 7 significant against layer contrasts at 0.07.
> The two arms were not comparable. The markers were scored over the whole brain against a null
> that shuffled the coupling, and the contrasts over the 1,768 cortical parcels of Schaefer-400
> against a null that rotates the mouse input. Re-scored like for like the markers give 0.072 with
> 3 of 7 significant, which sits inside the range of the contrasts they were supposed to exceed,
> and the dissociation does not survive. The manuscript deleted the claim and removed Fig. 3e.
> The as-published arm reproduces `hodge_2019_layer_markers.json` at 0.22819, which is what
> verifies the re-scoring. Log: `outputs/logs/hodge_markers_like_for_like.json`.

## 4 · OTTER versus TransBrain

Both methods scored as distributions over the same 127-region Brainnetome atlas, on the same 24
literature homologue pairs, using TransBrain's own atlas and curation.

| | OTTER | TransBrain |
|---|---:|---:|
| region-level AUROC | 0.83 | 0.84 |
| mass on the correct region | 0.21 | 0.07 |
| prediction sharpness (effective targets) | ~6 | ~60 |
| gradient translation (101 BN regions) | 0.56 | 0.52 |
| round-trip fidelity (52 matched regions) | 0.86–0.97 | 0.82–0.89 |
| spatial resolution | 2,094 parcels | ~127 regions |

The accuracy difference is not significant (paired Wilcoxon p = 0.36), so the two are level there.
Across the seven capability axes compared, OTTER leads on six and is level on the seventh. They are
probably better read as complementary instruments, region-level phenotype transfer against
calibrated whole-brain correspondence, than as competitors.

## 5 · Where the mouse cannot reconstruct human connectivity

Reconstruction accuracy asks how well each human parcel's connectivity fingerprint is rebuilt by
routing mouse connectivity through π.

```
pihat = pi / pi.sum(0);  pred = pihat.T @ Mfc @ pihat;  accuracy[j] = pearson(pred[j], Hfc[j])
```

> This replaced a measure that did not work. The earlier metric was the total mouse mass a parcel
> received, `log10(pi.sum(0))`, which is dominated by how spatially isolated a parcel is rather
> than by whether the mouse can account for it, and whose tail is set by the entropic
> regularisation. Every result derived from it is withdrawn, including the 6.7 log-unit
> sensorimotor–association gap (0.68 at spin p = 0.286 on the canonical coupling) and the
> uncovered-parcel percentages, which were threshold-dependent.

Reconstruction accuracy runs high over sensorimotor, auditory and visual territory and low over
prefrontal and lateral temporal cortex. One central visual parcel is rebuilt at r = 0.77, one
dorsolateral prefrontal parcel at r = 0.07. Across 1,824 cortical parcels the mean is r = 0.45, and
the measure agrees between homotopic regions across hemispheres at ρ = 0.36, so single-parcel
values are illustrative and the analyses below are conducted at tertile or network level.

### It tracks cortical expansion

Six of seven published maps clear a spin null:

| map | ρ | spin p |
|---|---:|---:|
| macaque→human expansion (Hill 2010) | −0.47 | 0.003 |
| mouse→human expansion (Xu 2020) | −0.32 | 0.001 |
| sensorimotor–association axis (Sydnor 2021) | −0.33 | 0.017 |
| principal FC gradient (Margulies 2016) | −0.28 | 0.017 |
| postnatal developmental expansion (Hill 2010) | −0.25 | 0.050 |
| mouse–human FC homology (Xu 2020) | +0.40 | 0.001 |
| T1w:T2w myelin (HCP) | +0.24 | 0.106 |

Splitting cortex by the sensorimotor–association axis, the association tertile sits at 0.40 against
0.50 for sensorimotor (Δ = 0.10, Cohen's d = 0.81, spin p = 0.010). The distributions overlap; what
separates them is the floor, which reaches 0.07 in association cortex and 0.22 in sensorimotor
cortex.

### The deficit is connectional rather than molecular

Control B, covering dorsolateral and rostrolateral prefrontal cortex, is the only network
significantly below the cortical mean (−0.69 SD, spin p = 0.006), while transcriptomic similarity
to mouse over the same parcels is flat (−0.18 SD, p = 0.39). Human dorsolateral prefrontal cortex
remains molecularly mammalian while having lost its connectional counterpart, so the species
difference is a reorganisation of connections rather than a replacement of tissue.

### A falsification control

Mouse medial-frontal cortex has no granular prefrontal homologue, and π does not manufacture one.
Of the mass it sends to the human brain, 32 % arrives in mid-cingulate cortex, 18 % in premotor
cortex and 11 % in medial prefrontal cortex, while 0.015 % reaches dorsolateral prefrontal cortex,
indistinguishable from the 0.026 % expected under a permuted coupling.

## 6 · Translating a mouse experiment

Routing a mouse anterior-insula optogenetic activation map through π gives a prediction that peaks
over anterior insula and ventral-attention cortex and is lowest in visual cortex. By Yeo-17
network, SalVentAttnB ranks first (+1.02 SD) and VisCent last (−1.30 SD). Salience enrichment
relative to the rest of cortex is +0.86 SD, against a permuted-π null at p = 0.001.

Against TransBrain on the same 1,635 parcels with matched permutation counts, OTTER reaches
+0.87 SD (p = 0.016) and TransBrain +0.28 SD (p = 0.228). OTTER ranks SalVentAttnB first of 16
networks; TransBrain ranks it eighth.

Cortical atrophy patterns from five mouse autism models route to different networks rather than
differing in severity, with salience enrichment from +0.15 (Dvl1) to −0.39 (Slc6a4). No null is
attached to the between-model comparison, and none should be read into the ordering.

### Reconstruction accuracy does not resolve disorders

Because §5 localises territory the mouse cannot reconstruct, it is natural to ask whether that map
predicts where a human disorder falls beyond the mouse's reach. It does not. Correlating
reconstruction accuracy with case-control cortical-thickness effect sizes across the
Desikan-Killiany atlas is null for all seven ENIGMA maps tested (minimum spin p = 0.15), and
weighting each disorder's thinning burden by reachability is null for all six disorders under the
same null (minimum p = 0.13, in autism).

The test detects a hierarchy-aligned effect when one is present. Run identically, the myelin
hierarchy map flags bipolar disorder (p = 0.028) and major depression (p = 0.011) in these same
data. Reconstruction accuracy carries no disorder-specific information at this resolution.

> An earlier draft reported the opposite, correlating mass-coverage with ENIGMA thinning and
> finding bipolar disorder (ρ = +0.64) and schizophrenia (ρ = +0.52) surviving FDR. That analysis
> used the retired mass-coverage metric on the retired coupling, and is withdrawn.

## Caveats

1. Parcel-exact recovery depends on the curation. Held out it collapses to roughly 10 %, while
   region-level recovery largely holds. The generalisation numbers to quote are the
   leave-one-region-out mean (0.74) and the curation-removed re-fit (0.90 → 0.73).
2. Parcel-level claims need the right tier. Trust parcel granularity in `anchored_and_validated`;
   use region granularity across the validated tiers.
3. The spatial scaffold is doing real work, and is itself fitted to the Garin pairs. No arm of the
   ablation is supervision-free, so OTTER is not unsupervised homology discovery.
4. Laminar structure does not translate. See §3, and do not read it out of π.
5. Reconstruction of association cortex is poor. The coupling reports the shortfall rather than
   hiding it, but a mouse model still cannot address phenotypes living there.
6. Correspondence is estimated between group-average connectomes, so it describes species rather
   than individuals.
7. Cerebellum and medulla are excluded from the parcellation. dlPFC homology is contested and
   opt-in only.

## How to use this map

1. Load the coupling with `load_pi()` and check the tier before trusting a prediction.
   `anchored_and_validated` supports parcel-level answers; either validated tier supports
   region-level; `low_evidence` is a hypothesis.
2. Ask whether your phenotype varies across the areal hierarchy. If it does, π will likely carry
   it. If it varies through cortical depth, it will not. §3 is the evidence.
3. Check reconstruction accuracy before translating into association cortex. Where it is low, the
   absence of a good counterpart is the finding rather than a failed query.
4. Spin-test every spatial correlation (`otter.eval.nulls.spin_null`) before reading it as
   significant. A permuted-π null is too lenient for a smooth map, and that is how two of the
   errors corrected on this page happened.
