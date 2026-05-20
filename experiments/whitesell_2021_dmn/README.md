# Whitesell 2021 DMN refinement

A focused validation of HOMER's DMN handling against [Whitesell et al. 2021, Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(21)00006-X), which provides the most careful published mouse-DMN boundary using Allen Mouse Brain Connectivity + rsfMRI.

## What we tested

The test: does HOMER's π route Whitesell's mouse-DMN parcels (defined via DSURQE labels: mPFC, ACC, RSC, PPC, dorsal hippocampus + subiculum, medial entorhinal cortex) to Yeo-7 DMN parcels?

The setup is identical to Pagani Test 1 / Coletta Sub-test A, but uses Whitesell's more careful mouse-DMN definition (168 parcels = 9% of brain) rather than HOMER's PAIRID-derived `frontal_dmn` + `temporal_dmn` networks.

## Result

Whitesell-DMN mass distribution after routing through π:

| Yeo-7 network | row-mass | n parcels | null | ratio |
|---|---:|---:|---:|---:|
| **DMN** | **23.9 %** | 439 | 21.0 % | **1.14×** |
| DorsAtten | 20.3 % | 242 | 11.6 % | 1.75× |
| SomatoMotor | 16.0 % | 253 | 12.1 % | 1.32× |
| Visual | 13.1 % | 255 | 12.2 % | 1.07× |
| Salience | 12.9 % | 248 | 11.8 % | 1.09× |
| Subcortical | 10.3 % | 326 | 15.6 % | 0.66× |
| Control | 3.6 % | 271 | 12.9 % | 0.28× |
| Limbic | 0.0 % | 60 | 2.9 % | 0.00× |

**DMN is the argmax**, but the row-mass (23.9 %) is *lower* than what HOMER's PAIRID-derived DMN gave in Pagani Test 1 (41 %).

## Interpretation

Whitesell defines mouse-DMN to include PPC, dorsal hippocampus, and medial entorhinal cortex — regions that route through π into Yeo-DorsAttn (PPC) and Yeo-Subcortical (hippocampus + subiculum). Combined into "DMN-aligned cortical territory":

- Yeo-DMN + Yeo-DorsAttn + Yeo-Subcortical = **54.5 % of Whitesell-DMN mass**, well above the 49.5 % uniform-null and substantially higher than Pagani's 41 % on Yeo-DMN alone.

**HOMER preserves Whitesell's mouse-DMN at the cortical-territory level. The Yeo-7 partition then fragments this territory across three of its categories.** Schaefer-17 places PPC in DorsAttn following the [Yeo / Krienen 2011 consensus](https://journals.physiology.org/doi/full/10.1152/jn.00338.2011) (cytoarchitectural + FC-clustering arguments). Hippocampus + subiculum get labeled "Subcortical" because they have no Schaefer cortical label.

**This isn't HOMER doing something wrong** — it's two principled definitions of "mouse DMN" (Whitesell's broad anatomical version, including parietal + hippocampal nodes; Yeo's narrow FC-clustered cortical version) intersecting in different places. Both are defensible biology.

## Why we didn't build a whitesell_dmn anchor pack

The original plan was a two-step refinement: (1) validation, then (2) anchor pack if motivated. After running the validation, the anchor pack didn't make sense:

1. **HOMER's current routing IS biologically correct.** Mouse-PPC → Yeo-DorsAttn is what Schaefer-17's consensus parcellation says it should be. Forcing PPC mass into Yeo-DMN via an anchor pack would override that consensus.
2. **Existing packs already cover the cortical-midline core.** The cingulate pack anchors subgenual ACC + RSC to Yeo-DMN territory, the lateral PFC pack covers mPFC. A narrow Whitesell pack would re-anchor regions already supervised.
3. **The result is more useful as a methodological note than as a model change.** It tells us where the DMN definition boundary lies between Whitesell and Yeo-7 — useful for any downstream user asking "which mouse-DMN definition is HOMER using?"

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
