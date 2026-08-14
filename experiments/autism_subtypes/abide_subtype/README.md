# ABIDE per-subject subtype scoring

Four scripts that score individual ABIDE subjects against human templates
obtained by translating the mouse hyper- and hypo-connectivity subtype maps of
Pagani 2026 through OTTER's coupling π.

The ABIDE data are distributed by the 1000 Functional Connectomes Project and
are available after registration. The outputs of these scripts are not
distributed with this repository. Reproducing them requires downloading the data
and running the scripts.

## Data

`nilearn.datasets.fetch_abide_pcp` supplies the preprocessed derivatives. The
scripts use the CPAC pipeline and the `rois_aal` derivative, which gives AAL-116
regional timeseries. The download is 3 to 8 GB and nilearn caches it, so it
happens once. `nilearn.datasets.fetch_atlas_aal` supplies the parcel definitions
used to align those timeseries to OTTER's 2,094 human parcels by nearest MNI
centroid.

## Scripts

### `abide_subtype_prediction.py`

Routes the mouse 9×9 network perturbation matrices (Pagani 2026, ED Fig 1)
through π to give one human template per mouse subtype, and takes their
difference (hyper − hypo) as a subtype-contrast template. For each subject the
script computes a per-parcel mean absolute connectivity profile, subtracts the
site-matched control mean, and projects the residual onto the contrast template.
The score is tested for a case-control difference (Mann-Whitney U, Cliff's δ)
and for bimodality within the case group (one- versus two-component Gaussian
mixture, AIC and BIC).

Writes `outputs/logs/autism_subtypes_abide.json` with the summary statistics and
`outputs/logs/abide_per_subject_scores.csv` with the per-subject scores.

```bash
# subset run
PYTHONPATH=src python experiments/autism_subtypes/abide_subtype/abide_subtype_prediction.py --n-subjects 50
# full run
PYTHONPATH=src python experiments/autism_subtypes/abide_subtype/abide_subtype_prediction.py
# alternative cache location
PYTHONPATH=src python experiments/autism_subtypes/abide_subtype/abide_subtype_prediction.py --abide-data-dir /path/to/abide_cache
```

The first run takes one to two hours once the download has finished. Later runs
take tens of minutes.

### `05_abide_otter_subtyping.py`

Applies the discrete classification rule of Pagani 2026, under which a subject
is hypo if regional global connectivity falls below −1 s.d. within the hypo mask
and hyper if it exceeds +1 s.d. within the hyper mask, under two definitions of
the human masks. One is the set of human regions π routes the prominent mouse
regions to, the other is the set of human regions carrying the same names as
those mouse regions. The script reports both subtypings and their agreement.

Requires `experiments/pagani_2026_per_model/04_otter_human_masks.py` to have run
first, which writes `outputs/logs/pagani_otter_human_masks.json`. Writes
`outputs/logs/abide_otter_subtyping.json`.

```bash
PYTHONPATH=src python experiments/autism_subtypes/abide_subtype/05_abide_otter_subtyping.py --abide-data-dir /path/to/abide_cache
```

### `06_continuous_subtype_score.py`

Replaces the ±1 s.d. threshold with a continuous coordinate. The contrast
between the human hyper and hypo coupling maps gives a per-region weight, and a
subject's position on the axis is the weighted sum of their z-scored regional
global connectivity. The script compares the axis against the discrete labels
from `05`, compares cases with controls, and correlates the axis with symptom
severity.

Same prerequisites as `05`. Writes `outputs/logs/abide_continuous_subtype.json`.

```bash
PYTHONPATH=src python experiments/autism_subtypes/abide_subtype/06_continuous_subtype_score.py --abide-data-dir /path/to/abide_cache
```

### `plot_abide.py`

Reads the two outputs of `abide_subtype_prediction.py` and writes
`outputs/figures/autism_subtypes_abide.png`.

## Options

`--abide-data-dir` sets the nilearn cache directory and defaults to
`~/abide_cache`. `--pipeline` selects the preprocessing pipeline and defaults to
`cpac`. `--n-subjects`, in `abide_subtype_prediction.py`, caps the number of
subjects loaded.

## Limitations

The per-subject perturbation feature is a mean absolute connectivity per parcel,
coarser than the per-cell contrast against controls used by Pagani. The AAL to
OTTER alignment is by nearest centroid, where a volume-weighted assignment would
be more accurate. Controls are matched by site, and age, sex and head motion are
not regressed out.
