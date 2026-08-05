# ENIGMA cross-disorder spatial validation

Two-phase pipeline: Phase 1 generates OTTER's per-disorder predicted human spatial patterns; Phase 2 compares against ENIGMA's published cross-disorder cortical-thickness Cohen's d maps.

## Why this experiment

The Pagani-based cross-disease specificity check (autism vs schizophrenia vs ADHD vs bipolar gene sets routed through π) found at network resolution that **OTTER's predictions are not disorder-specific**: all 4 disorders gave r ≈ +0.4 against Pagani's observed autism Δ. This experiment sharpens that test by:

1. Generating OTTER's predicted human spatial patterns at full **parcel resolution** (2,094 parcels) for each disorder
2. Computing the **cross-disorder correlation matrix** to quantify how similar OTTER's predictions are
3. (Phase 2) Comparing against ENIGMA's actual disease cortical-thickness maps at Desikan-Killiany region resolution

## Phase 1. In-sandbox result

For each disorder in Pagani MOESM4 (autism) + MOESM5 (bipolar, schizophrenia, ADHD), translate its gene set through π → predicted human-parcel spatial pattern.

Cross-disorder correlation matrix (mean off-diagonal r):

On the canonical coupling (`pi_canonical.npy`, sha256 `bb4cae00…`):

| OTTER-disorder pair | Pearson r |
|---|---:|
| autism ↔ schizophrenia | +0.999 |
| autism ↔ bipolar | +0.993 |
| autism ↔ ADHD | +0.993 |
| bipolar ↔ schizophrenia | +0.992 |
| bipolar ↔ ADHD | +0.985 |
| schizophrenia ↔ ADHD | +0.994 |
| **Mean off-diagonal** | **+0.993** |

**OTTER's per-disorder predictions are essentially identical at parcel resolution.** This confirms the cross-disease specificity finding from earlier, and is even stronger at parcel level (mean off-diagonal +0.993) than at network level. OTTER captures a *generic brain-disorder spatial geometry* rather than disorder-specific signals.

Implication: psoriasis (skin disease, non-brain) was excluded because only 2 of its 18 genes overlap with OTTER's panel, so the non-brain control cannot be tested directly. The consistency across the 4 brain disorders that can be tested is nonetheless strong evidence that OTTER doesn't discriminate at this level.

## Phase 2. ENIGMA comparison (needs external data)

To test how OTTER's generic prediction matches ENIGMA's actual observed cortical-thickness Cohen's d maps per disorder, you need to download the ENIGMA Toolbox summary statistics:

```bash
git clone https://github.com/MICA-MNI/ENIGMA.git /tmp/ENIGMA
mkdir -p otter/data_external/enigma
cp /tmp/ENIGMA/enigmatoolbox/datasets/summary_statistics/cortical_thickness_*.csv \
   otter/data_external/enigma/

# Then run Phase 2
PYTHONPATH=src python experiments/enigma_cross_disorder/03_enigma_comparison.py
```

The script:
1. Loads ENIGMA Cohen's d per Desikan-Killiany region for each disorder
2. Aggregates OTTER's parcel-level predictions to DK regions via nearest-MNI-centroid mapping
3. Computes Pearson r between OTTER-predicted and ENIGMA-observed per-disorder
4. Reports diagonal (OTTER-X vs ENIGMA-X) vs off-diagonal (OTTER-X vs ENIGMA-Y) correlation

**Expected outcome given Phase 1's result**: OTTER-predicted patterns will correlate similarly with every ENIGMA disease map (diagonal ≈ off-diagonal), because the predictions themselves don't differ between disorders. The question Phase 2 answers is a different one: **how strongly does OTTER's generic brain-disorder geometry match what is observed across psychiatric ENIGMA disorders?** A strong overall correlation would mean OTTER captures the shared "psychiatric perturbation territory" in human cortex.

## Files

| File | What |
|---|---|
| `01_per_disorder_prediction.py` | Phase 1, generate OTTER per-disorder predictions + cross-disorder correlation matrix |
| `02_plot_phase1.py` | Phase 1 figure |
| `03_enigma_comparison.py` | Phase 2 scaffold, runs once ENIGMA CSVs are in `data_external/enigma/` |
| `README.md` | This file |

## Reproduce

**Phase 1** (no external data needed):
```bash
PYTHONPATH=src python experiments/enigma_cross_disorder/01_per_disorder_prediction.py
PYTHONPATH=src python experiments/enigma_cross_disorder/02_plot_phase1.py
```

**Phase 2** (requires ENIGMA Toolbox CSVs):
```bash
# After cloning ENIGMA + copying CSVs as above
PYTHONPATH=src python experiments/enigma_cross_disorder/03_enigma_comparison.py
```

Outputs:
- `outputs/logs/enigma_phase1_per_disorder.json`
- `outputs/logs/enigma_phase2_comparison.json` (after Phase 2 run)
- `outputs/figures/enigma_phase1_per_disorder.png`
- `outputs/coupling/per_disorder_predictions.npz`


## Disorder-unique + transdiagnostic (2026-06-19), `04_disorder_unique.py`, `05_transdiagnostic.py`

**Is the r=0.988 "no specificity" just gene-set overlap?** No. `04_disorder_unique.py`
strips each disorder to the genes unique to it and re-routes. The non-autism sets turn out
to be essentially *nested* in the 1,713-gene autism set, so we use a pairwise
relative-unique test (genes in A-not-B vs B-not-A): even **fully disjoint** sets
(bipolar-only 26 genes vs SCZ-only 447 genes) still route to near-identical human maps
(r=+0.98). The shared psychiatric geometry is therefore **robust rather than an overlap
artifact**.

**Does that shared geometry match real disease maps?** `05_transdiagnostic.py` compares
OTTER's generic predicted map to ENIGMA observed cortical-thickness Cohen's d (DK-68),
including a transdiagnostic average across ASD/SCZ/BD/ADHD and a spin null over DK
centroids. Result: r=−0.27 vs the transdiagnostic average, **spin p=0.24 (n.s.)**, and
n.s. for every individual disorder and the held-out MDD/OCD. So OTTER's shared geometry
is a property of the gene→π routing and does **not** align with the observed ENIGMA
thickness signature beyond spatial autocorrelation. Negative result against external
maps. (ENIGMA CSVs: ENIGMA Toolbox `summary_statistics`, staged in
`data_external/enigma/`.) Logs: `enigma_disorder_unique.json`, `enigma_transdiagnostic.json`.
