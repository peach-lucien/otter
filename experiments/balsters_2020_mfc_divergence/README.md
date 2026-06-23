# Balsters 2020, rodent MFC divergence (falsification test)

A **falsification** test: it states, with a direction, where a faithful
mouse↔human mapping should *not* send mouse medial frontal cortex, and
checks whether HOMER's π obeys.

## Why this experiment

[Balsters, Zerbi, Sallet, Wenderoth & Mars 2020 (PNAS)](https://doi.org/10.1073/pnas.2003181117),
"Divergence of rodent and primate medial frontal cortex functional
connectivity", compared whole-brain FC of the medial frontal cortex across
rodent, marmoset and human. Their data-backed conclusion: rodent MFC does
**not** correspond to primate **dorsolateral PFC** (contradicting the common
proposal that rat MFC is the analogue of primate lateral PFC); its
connectivity instead most resembles **premotor** cortex.

That is a specific, directional, published "where it should NOT go"
prediction. Most HOMER validations test for a *positive* signal; this one
tests whether HOMER avoids a *wrong* answer:

- **PASS**, mouse MFC routes to human medial-frontal / cingulate / premotor cortex and avoids dlPFC.
- **FAIL**, mouse MFC routes confidently onto human dlPFC (BA9/46).

HOMER already makes a falsifiable design choice here. Its Garin point anchor
pairs mouse mPFC with human *medial* frontal cortex, and the contested
mouse-Prelimbic ↔ human-dlPFC homology (Carlén 2017 vs Preuss 1995) ships as
the **opt-in** `lateral_pfc` pack, not in the recommended π. Balsters 2020
is independent FC evidence adjudicating that choice.

*Species note:* Balsters used rat + marmoset + human; HOMER is mouse + human.
Rodent MFC (anterior cingulate + prelimbic + infralimbic) is the comparable
structure. The test compares HOMER's π against Balsters' published
*directional conclusion*, not their FC matrices, the rat/mouse and
marmoset/human mismatches make re-routing their data unjustified.

## Result

**VERDICT: PASS. Balsters-consistent.** Three panels
(`outputs/figures/balsters_2020_mfc_divergence.png`):

**1. Where π routes mouse MFC.** Of the coupling mass leaving HOMER's 39
mouse rodent-MFC parcels (ACAd/ACAv/PL/ILA), **0.0 %** reaches human dlPFC
an enrichment of **×0.0** against the permuted-π null (empirical p = 0.99,
i.e. observed mass is at the *bottom* of the null distribution). The
Balsters-consistent territories are strongly favoured: medial PFC ×10.1,
premotor ×6.0, mid-cingulate ×4.3 (all p ≤ 0.025). Of the 39 MFC parcels,
**0** have their top-1 human partner in dlPFC; 7 land in medial PFC, 6 in
premotor, 2 in mid-cingulate.

**2. The contested anchor is correctly quarantined.** Mouse-MFC mass onto
human dlPFC is 0.0 % under the Garin-only baseline and 0.0 % under the
recommended π. It only appears, jumping to **23.1 %** (and 18.2 % for
Prelimbic specifically), when the opt-in `lateral_pfc` pack's contested
Prelimbic→dlPFC anchor is forced in. Balsters 2020 is the independent FC
evidence that this anchor encodes a homology the data argues against, and
supports HOMER's decision to keep it opt-in.

**3, dlPFC is a low-homology territory.** Human dlPFC receives only 0.6 %
of *all* mouse→human coupling mass while occupying 1.1 % of human parcels
under-represented relative to chance. HOMER finds no confident rodent
homologue of dorsolateral PFC from anywhere in the mouse brain, consistent
with the evolutionary expansion of human lateral PFC.

| Test (recommended π) | Result |
|---|---|
| mouse MFC → human dlPFC | **0.0 %** mass, **0/39** argmax, enrichment ×0.0 |
| mouse MFC → premotor / medial PFC / mid-cingulate | 16.3 / 17.9 / 5.1 %, enrichment ×6–10 |
| dlPFC mass only with `+lateral_pfc` pack | 23.1 % (contested anchor) |

## Method

1. Mouse rodent-MFC = 1,864-parcel rows with Allen acronym in
   {ACAd, ACAv, PL, ILA} (39 parcels; Prelimbic alone = 11).
2. Human target ROIs = bilateral MNI spheres, each human parcel assigned to
   its nearest ROI within radius: dlPFC (±40, 25, 35) r=12, the lateral_pfc
   pack's own BA9/46 centroid; premotor (±28, 0, 54) r=14; medial PFC
   (±7, 34, 22) r=16; mid-cingulate (±5, 8, 40) r=14.
3. For each π, compute the fraction of mouse-MFC row mass landing in each
   ROI, and the top-1 (argmax) human partner of every MFC parcel.
4. Permuted-π null (200 row shuffles) → per-ROI enrichment and empirical p.
5. Contrast across three couplings: Garin-only baseline, recommended
   (`with_all_packs`), and `with_lateral_pfc` (adds the contested anchor).

## Files

| File | What |
|---|---|
| `01_mfc_divergence.py` | Route mouse MFC through π, mass/argmax by human ROI, permuted-π null, three-coupling contrast |
| `02_plot.py` | 3-panel figure |
| `README.md` | This file |

## Reproduce

```bash
PYTHONPATH=src python experiments/balsters_2020_mfc_divergence/01_mfc_divergence.py
PYTHONPATH=src python experiments/balsters_2020_mfc_divergence/02_plot.py
```

Outputs:
- `outputs/logs/balsters_2020_mfc_divergence.json`
- `outputs/figures/balsters_2020_mfc_divergence.png`

## Showcase notebook

See [`notebooks/13_balsters_2020_mfc_divergence.ipynb`](../../notebooks/13_balsters_2020_mfc_divergence.ipynb).
