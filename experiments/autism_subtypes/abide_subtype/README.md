# ABIDE per-subject HOMER-template subtype scoring (PAGANI-B)

## What this tests

Pagani 2026's claim 1 is that ASD subjects cluster into hyper- and hypo-connected functional-connectivity subtypes. They derive those subtypes by an in-house clustering pipeline on FC perturbation features.

We don't re-implement their clustering. Instead we ask a sharper question that uses HOMER's contribution directly: **does HOMER's translation of mouse subtype patterns produce a feature that distinguishes ASD from controls at the individual level?** And: **does that feature reveal hyper-vs-hypo bimodality within ASD?**

If yes — HOMER's quantitative cross-species translation can serve as an automatic, biologically-grounded subtype classifier for human ASD, replacing the name-based bridge Pagani uses with a structural one.

## Pipeline

1. **Build HOMER human templates.** Translate Pagani's mouse 9×9 hypo and hyper perturbation matrices (ED Fig 1) through π → 2,094-parcel human templates. Take Δ = hyper − hypo as the *subtype-contrast* template.

2. **Fetch ABIDE preprocessed FC.** Via `nilearn.datasets.fetch_abide_pcp` — uses CPAC pipeline, CC400 parcellation, ~1,000 ASD + ~1,100 control subjects across ~24 sites.

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

- `outputs/logs/autism_subtypes_abide.json` — summary stats (Mann-Whitney p, Cliff's δ, GMM bimodality verdict, n_asd/n_ctrl)
- `outputs/logs/abide_per_subject_scores.csv` — every subject's HOMER score + phenotype (for downstream plotting / replication)

## Expected outcome (priors)

If HOMER's translation is biologically meaningful:
- ASD score mean should differ from control (Mann-Whitney p < 0.05; Cliff's δ ≠ 0).
- Within-ASD GMM should marginally prefer 2-component fit (BIC verdict).

If neither: HOMER's translation captures cross-species pattern at the network-aggregate level (Test 2c) but not at the subject-resolution noise level — useful information either way for the manuscript.

## Caveats

- **Coarse FC perturbation metric.** We use mean-abs-FC-per-parcel as the subject perturbation feature; Pagani uses a more sophisticated per-cell vs-control contrast. Our coarser version may underestimate true effects.
- **Parcellation translation.** CC400 → HOMER nearest-centroid mapping introduces error. A volume-weighted mapping would be more rigorous (~1 day to implement properly).
- **Site / age regression.** We site-match controls but don't formally regress out age, sex, motion. Standard ABIDE analyses do; this is a screening test.

If results look promising (Cliff's δ > 0.1, p < 0.05) the natural next iteration is to harden these three caveats.

## What HOMER contributes

The HOMER score is **the only feature in this pipeline derived from mouse data**. Every other feature is human-only (ABIDE FC, Craddock parcellation). If HOMER's translation of mouse subtype patterns generates a feature that classifies human ASD vs controls, that's a direct demonstration that the cross-species mouse model literature carries clinically-relevant signal — operationalized through HOMER's π rather than through the name-based bridge Pagani uses.
