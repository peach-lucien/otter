# Parkinson stage maps

The analysis in `../08_pd_stage_progression.py` uses regional cortical-thickness
effect sizes from:

Laansma MA et al. International Multicenter Analysis of Brain Structure Across
Clinical Stages of Parkinson's Disease. *Movement Disorders* 36, 2583–2594
(2021). DOI: `10.1002/mds.28706`.

The expected local inputs are:

| Local file | Hoehn–Yahr group |
|---|---|
| `data_external/enigma/cortical_thickness_parkinsons_HY1.csv` | Stage 1 |
| `data_external/enigma/cortical_thickness_parkinsons_HY2.csv` | Stage 2 |
| `data_external/enigma/cortical_thickness_parkinsons_HY3.csv` | Stage 3 |
| `data_external/enigma/cortical_thickness_parkinsons_HY4and5.csv` | Stages 4/5 |

Each table contains bilateral Desikan–Killiany regional Cohen's *d* values.
The analysis negates the published effect so that larger values denote greater
cortical thinning, z-standardises regions within stage and records a SHA-256
digest for every source table. The analysis evaluates an investigator-defined
directional contrast between interoceptive and primary-motor mouse cortex.
