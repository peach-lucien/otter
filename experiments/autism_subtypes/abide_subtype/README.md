# ABIDE per-subject HOMER-template subtype scoring (PAGANI-B)

## Corrected, Pagani-faithful subtyping (05/06), read this first

The original script below (`abide_subtype_prediction.py`) scores subjects against a
continuous Δ = hyper − hypo template. HOMER's *weak* (continuous-map) mode. Two
newer scripts implement Pagani's *actual* discrete classification and a continuous
extension, and supersede the framing of the older one:

- **`05_abide_homer_subtyping.py`** reproduces Pagani's exact ±1 s.d. classification
  on ABIDE, comparing HOMER-derived masks against name-matched masks head to head.
  **Result (ran 2026):** HOMER masks subtype **21.3%** of ASD, name-matched **22.3%**
  (Pagani report ~25%), with **93% label agreement**. So HOMER does *not* subtype more
  individuals, it ≈equals the name-matched bridge and thereby **validates** it
  (discrete homology survives; this is HOMER's strong mode). The ~78% unsubtyped is a
  *hard-threshold* bottleneck, not a mapping failure. (NB: our pipeline uses AAL-116,
  not Pagani's Schaefer-400+14, so absolute % isn't directly comparable; the
  HOMER-vs-name comparison is internally consistent.)

- **`06_continuous_subtype_score.py`** removes the hard threshold: every individual
  gets a continuous position on the HOMER hyper↔hypo axis (projection of z-scored
  regional global connectivity onto the hyper−hypo coupling contrast). This tests the
  dose-response Pagani's binary scheme structurally can't. **Result (ran 2026-06-19):
  NULL.** The axis *is* a valid construct (it orders the hard labels correctly:
  hard-hyper mean +0.110 > hard-hypo +0.025), but it carries **no diagnostic signal**
  (ASD vs control Mann-Whitney p=0.97) and **no severity dose-response** (axis vs ADOS:
  every subscale |ρ|≤0.11, all n.s.; closest ADOS_SOCIAL ρ=−0.11, p=0.078). This is
  consistent with HOMER's established dichotomy, discrete correspondence survives,
  continuous/graded translation does not, and with Pagani themselves never having
  shown a continuous ADOS dose-response. **Caveat (F-015):** the axis inherits π's
  uneven human coverage (masks lean Subcortical/Salience/DMN), so the claim is
  "no detectable continuous severity signal under the current coupling," not "the
  continuum is flat." Output: `outputs/logs/abide_continuous_subtype.json`.

**Bottom line for the manuscript:** the per-individual continuous subtype readout is a
clean negative; the discrete HOMER↔name-match equivalence (05) is the positive result
worth reporting. See `../../pagani_2026_per_model/README.md` for the mouse-side story.

## What this tests

Pagani 2026's claim 1 is that ASD subjects cluster into hyper- and hypo-connected functional-connectivity subtypes. They derive those subtypes by an in-house clustering pipeline on FC perturbation features.

We don't re-implement their clustering. Instead we ask a sharper question that uses HOMER's contribution directly: **does HOMER's translation of mouse subtype patterns produce a feature that distinguishes ASD from controls at the individual level?** And: **does that feature reveal hyper-vs-hypo bimodality within ASD?**

If yes. HOMER's quantitative cross-species translation can serve as an automatic, biologically-grounded subtype classifier for human ASD, replacing the name-based bridge Pagani uses with a structural one.

## Pipeline

1. **Build HOMER human templates.** Translate Pagani's mouse 9×9 hypo and hyper perturbation matrices (ED Fig 1) through π → 2,094-parcel human templates. Take Δ = hyper − hypo as the *subtype-contrast* template.

2. **Fetch ABIDE preprocessed FC.** Via `nilearn.datasets.fetch_abide_pcp`, uses CPAC pipeline, CC400 parcellation, ~1,000 ASD + ~1,100 control subjects across ~24 sites.

3. **Map CC400 → HOMER's 2,094 parcels.** Nearest-MNI-centroid (Craddock 2012 atlas, 400-cluster level).

4. **Per-subject perturbation profile.** For each subject: compute per-parcel mean-abs-FC strength, subtract site-matched control mean. This is a coarse but defensible approximation of Pagani's subject-level FC perturbation map.

5. **Score each subject.** dot-product of subject perturbation with the HOMER Δ template (z-scored, mean-normalized).

6. **Tests.**
   - ASD vs control on score → Mann-Whitney U + Cliff's δ.
   - Within-ASD bimodality → 1-vs-2 component Gaussian mixture (AIC + BIC).

## Disk + time

- ABIDE preprocessed CC400 timeseries: **~3–8 GB**, one-time download via nilearn.
- Craddock atlas: ~10 MB.
- Wall-clock for first run: **~1–2 hours after download finishes**.
- Subsequent runs: ~10–30 minutes (nilearn caches everything).

## How to run

### Smoke test (50 subjects, ~5–10 min wall-clock after the first 1–2 GB of ABIDE downloads)

```bash
cd <homer-repo-root>
PYTHONPATH=src python experiments/autism_subtypes/abide_subtype/abide_subtype_prediction.py \
    --n-subjects 50
```

### Full run

```bash
PYTHONPATH=src python experiments/autism_subtypes/abide_subtype/abide_subtype_prediction.py
```

### Custom cache dir (if home directory is small)

```bash
PYTHONPATH=src python ... --abide-data-dir /path/to/big/disk/abide_cache
```

## Outputs

- `outputs/logs/autism_subtypes_abide.json`, summary stats (Mann-Whitney p, Cliff's δ, GMM bimodality verdict, n_asd/n_ctrl)
- `outputs/logs/abide_per_subject_scores.csv`, every subject's HOMER score + phenotype (for downstream plotting / replication)

## Expected outcome (priors)

If HOMER's translation is biologically meaningful:
- ASD score mean should differ from control (Mann-Whitney p < 0.05; Cliff's δ ≠ 0).
- Within-ASD GMM should marginally prefer 2-component fit (BIC verdict).

If neither: HOMER's translation captures cross-species pattern at the network-aggregate level (Test 2c) but not at the subject-resolution noise level, useful information either way for the manuscript.

## Caveats

- **Coarse FC perturbation metric.** We use mean-abs-FC-per-parcel as the subject perturbation feature; Pagani uses a more sophisticated per-cell vs-control contrast. Our coarser version may underestimate true effects.
- **Parcellation translation.** CC400 → HOMER nearest-centroid mapping introduces error. A volume-weighted mapping would be more rigorous (~1 day to implement properly).
- **Site / age regression.** We site-match controls but don't formally regress out age, sex, motion. Standard ABIDE analyses do; this is a screening test.

If results look promising (Cliff's δ > 0.1, p < 0.05) the natural next iteration is to harden these three caveats.

## What HOMER contributes

The HOMER score is **the only feature in this pipeline derived from mouse data**. Every other feature is human-only (ABIDE FC, Craddock parcellation). If HOMER's translation of mouse subtype patterns generates a feature that classifies human ASD vs controls, that's a direct demonstration that the cross-species mouse model literature carries clinically-relevant signal, operationalized through HOMER's π rather than through the name-based bridge Pagani uses.
