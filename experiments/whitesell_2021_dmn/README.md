# Whitesell 2021 DMN refinement

A focused validation of HOMER's DMN handling against [Whitesell et al. 2021, Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(21)00006-X), which provides the most careful published mouse-DMN boundary using Allen Mouse Brain Connectivity + rsfMRI.

## What we tested

The test: does HOMER's π route Whitesell's mouse-DMN parcels (defined via DSURQE labels: mPFC, ACC, RSC, PPC, dorsal hippocampus + subiculum, medial entorhinal cortex) to Yeo-7 DMN parcels?

The setup is identical to Pagani Test 1 / Coletta Sub-test A, but uses Whitesell's more careful mouse-DMN definition (176 parcels = 9.4 % of brain) rather than HOMER's PAIRID-derived `frontal_dmn` + `temporal_dmn` networks.

## Result

On the canonical coupling (`pi_canonical.npy`, sha256 `bb4cae00…`), Whitesell-DMN mass distribution after routing through π:

| Yeo-7 network | row-mass | n parcels | null | ratio |
|---|---:|---:|---:|---:|
| **DMN** | **38.0 %** | 439 | 21.0 % | **1.81×** |
| Subcortical | 19.3 % | 326 | 15.6 % | 1.24× |
| Salience | 13.2 % | 248 | 11.8 % | 1.12× |
| Limbic | 11.8 % | 60 | 2.9 % | 4.13× |
| Visual | 10.4 % | 255 | 12.2 % | 0.86× |
| Control | 4.3 % | 271 | 12.9 % | 0.33× |
| DorsAtten | 1.9 % | 242 | 11.6 % | 0.17× |
| SomatoMotor | 0.9 % | 253 | 12.1 % | 0.07× |

**DMN is the argmax at 1.81× the uniform null.**

### Comparison to the PAIRID-derived DMN baseline (corrected 2026-07-18)

The baseline for this comparison is Pagani Test 1 (`outputs/logs/autism_subtypes_network_crossval.json`). That log was still on the retired coupling `pi_fc_plus_SC_with_all_packs.npy` when the comparison above was first written, so it was **not like-for-like**: a canonical-π Whitesell number (38.0 %) was being compared against a retired-π Pagani number (38.2 %), which is how the earlier "same range" reading arose. The baseline has now been re-run on `pi_canonical.npy` (same sha256 as this experiment).

| mouse-DMN definition | row-mass into Yeo-DMN | ratio vs parcel-count null (21.0 %) | ratio vs π-mass null (9.8 %) |
|---|---:|---:|---:|
| PAIRID-derived (Pagani Test 1), **canonical π** | 26.0 % | 1.24× | 2.64× |
| PAIRID-derived (Pagani Test 1), *retired π (superseded)* | *38.2 %* | – | *2.19×* |
| **Whitesell 2021, canonical π** | **38.0 %** | **1.81×** | **3.86×** |

Both experiments use the same 439-parcel Yeo-DMN mask and the same row-normalised π-mass statistic, so the row-mass column is directly comparable. The two scripts report different nulls (Whitesell uses a parcel-count uniform null, Pagani Test 1 a π-mass-weighted null); both are shown above, and the ordering is the same under either.

**Corrected verdict: refining the mouse-DMN parcel set to Whitesell's definition does sharpen the DMN→DMN correspondence** (38.0 % vs 26.0 %, 1.81× vs 1.24× over the parcel-count null). The earlier "neither sharpens nor degrades" reading was an artefact of comparing across two different couplings.

## Interpretation

> **Stale-coupling warning (2026-07-18).** Everything below this line was written against the retired pre-warp coupling `pi_fc_plus_SC_with_all_packs.npy`, on which DorsAttn took 20.3 % of Whitesell-DMN mass and Limbic took 0.0 %. Under the canonical coupling those are 1.9 % and 11.8 %, so the "fragmented across DMN + DorsAttn + Subcortical" reading no longer follows from the numbers. The DMN + DorsAttn + Subcortical sum is now 59.2 % against a 48.2 % uniform null. This section needs rewriting against the canonical result; it has been left in place rather than silently re-narrated.

Whitesell defines mouse-DMN to include PPC, dorsal hippocampus, and medial entorhinal cortex, regions that route through π into Yeo-DorsAttn (PPC) and Yeo-Subcortical (hippocampus + subiculum). Combined into "DMN-aligned cortical territory":

- ~~Yeo-DMN + Yeo-DorsAttn + Yeo-Subcortical = **54.5 % of Whitesell-DMN mass**, well above the 49.5 % uniform-null and substantially higher than Pagani's 41 % on Yeo-DMN alone.~~ *(retired-π numbers.)* On canonical π the same sum is **59.2 %** against a **48.2 %** parcel-count null, and the PAIRID-derived Pagani baseline on Yeo-DMN alone is **26.0 %**, not 41 %. Note that the DorsAttn term is now only 1.9 % (it was 20.3 % on the retired coupling), so the "PPC routes into Yeo-DorsAttn" premise below no longer holds and the three-way-fragmentation argument needs rebuilding rather than renumbering.

**HOMER preserves Whitesell's mouse-DMN at the cortical-territory level. The Yeo-7 partition then fragments this territory across three of its categories.** Schaefer-17 places PPC in DorsAttn following the [Yeo / Krienen 2011 consensus](https://journals.physiology.org/doi/full/10.1152/jn.00338.2011) (cytoarchitectural + FC-clustering arguments). Hippocampus + subiculum get labeled "Subcortical" because they have no Schaefer cortical label.

**This is not a HOMER error.** Two principled definitions of "mouse DMN" (Whitesell's broad anatomical version, including parietal + hippocampal nodes; Yeo's narrow FC-clustered cortical version) intersect in different places. Both are defensible biology.

## Why we didn't build a whitesell_dmn anchor pack

The original plan was a two-step refinement: (1) validation, then (2) anchor pack if motivated. The validation removed the case for the anchor pack:

1. **HOMER's current routing is biologically correct.** Mouse-PPC → Yeo-DorsAttn is what Schaefer-17's consensus parcellation says it should be. Forcing PPC mass into Yeo-DMN via an anchor pack would override that consensus.
2. **Existing packs already cover the cortical-midline core.** The cingulate pack anchors subgenual ACC + RSC to Yeo-DMN territory, and the lateral PFC pack covers mPFC. A narrow Whitesell pack would re-anchor regions already supervised.
3. **The result is more useful as a methodological note than as a model change.** It locates the DMN-definition boundary between Whitesell and Yeo-7, which is useful for any downstream user asking "which mouse-DMN definition is HOMER using?"

## Files

| File | What |
|---|---|
| `01_whitesell_dmn_refinement.py` | The validation: aggregate π for Whitesell-DMN parcels → Yeo-7 row-mass |
| `README.md` | This file |

## Reproduce

```bash
PYTHONPATH=src python experiments/whitesell_2021_dmn/01_whitesell_dmn_refinement.py
```

Output: `outputs/logs/whitesell_2021_dmn_refinement.json`.
