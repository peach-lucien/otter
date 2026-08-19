# ABIDE analyses

Two independent pipelines run on ABIDE. They use different derivatives and
answer different questions.

The **case-control pipeline** (`abide_ho_s0` to `abide_ho_s7`) uses the
Harvard-Oxford derivative and tests whether a mouse mutation pattern translated
through π separates autism cases from controls. This is the analysis reported in
the manuscript.

The **subtyping pipeline** (`abide_subtype_prediction.py`, `05`, `06`,
`plot_abide.py`) uses the AAL derivative and asks how many individuals the
coupling assigns to a hyper- or hypo-connectivity subtype, against the discrete
rule of Pagani 2026.

The sections below cover the subtyping pipeline first, then the case-control
pipeline.

The ABIDE data are distributed by the 1000 Functional Connectomes Project and
are available after registration. The outputs of these scripts are not
distributed with this repository. Reproducing them requires downloading the data
and running the scripts.

## Subtyping pipeline

### Data

`nilearn.datasets.fetch_abide_pcp` supplies the preprocessed derivatives. The
scripts use the CPAC pipeline and the `rois_aal` derivative, which gives AAL-116
regional timeseries. The download is 3 to 8 GB and nilearn caches it, so it
happens once. `nilearn.datasets.fetch_atlas_aal` supplies the parcel definitions
used to align those timeseries to OTTER's 2,094 human parcels by nearest MNI
centroid.

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

### Options

`--abide-data-dir` sets the nilearn cache directory and defaults to
`~/abide_cache`. `--pipeline` selects the preprocessing pipeline and defaults to
`cpac`. `--n-subjects`, in `abide_subtype_prediction.py`, caps the number of
subjects loaded.

### Limitations

The per-subject perturbation feature is a mean absolute connectivity per parcel,
coarser than the per-cell contrast against controls used by Pagani. The AAL to
OTTER alignment is by nearest centroid, where a volume-weighted assignment would
be more accurate. Controls are matched by site, and age, sex and head motion are
not regressed out.


---

## Case-control pipeline

Seven steps that test whether a mouse mutation pattern, translated through π,
separates autism cases from controls in ABIDE. The derivative is `rois_ho`,
Harvard-Oxford regional time series, rather than the AAL derivative used above.

Every step writes to `.scratch/abide_ho/` and reads the bundle built by step 0.
The bundle location is `data_external/abide_ho/` by default and can be moved
with `OTTER_ABIDE_BUNDLE`. The nilearn download cache is `~/abide_cache` by
default and can be moved with `OTTER_ABIDE_DIR`. Run every step from the
repository root.

### `abide_ho_s0_bundle.py`

Downloads the `rois_ho` derivative through nilearn, reads the Harvard-Oxford ROI
ids from each `.1D` header, and writes the connectivity bundle, the phenotype
table and the three Harvard-Oxford atlases the later steps need. The per-subject
feature is the mean absolute Fisher-z correlation of each ROI with every other
ROI. The grand matrix is the mean Fisher-z correlation matrix over subjects.
`qc_ok` is set from the ABIDE quality-control rater columns, and the number of
passing subjects is printed so it can be checked against the sample the analysis
reports.

The download is several GB and happens once. Nothing derived from it is
redistributed with this repository.

```bash
python experiments/autism_subtypes/abide_subtype/abide_ho_s0_bundle.py
```

### `abide_ho_s1_atlas.py`

Centroids of every Harvard-Oxford label in MNI mm, and the rule that maps a
hemisphere-split cortical label L to its ABIDE id, `((L + 1) // 2) * 100` plus 1
for odd L and 2 for even L.

### `abide_ho_s2_val.py`

Checks on the atlas to column correspondence: connectivity against Euclidean
distance, the rank of each cortical region's homotopic partner in its own
connectivity row against a random contralateral region, and the strongest
partners of the column that carries no atlas label.

### `abide_ho_s3_tpl.py`

Routes every mouse mutation pattern in the TransBrain 2025 table through π. Each
human parcel takes the mass-weighted mean of the mouse parcels reaching it. The
Magel2 template is cached in translated form as the worked example for step 4.

### `abide_ho_s4_main.py`

Scores every participant against the Magel2 template on two column mappings. The
`label_matched` arm assigns each ABIDE column the centroid of its own ROI id.
The `positional` arm stacks the bilateral cortical centroids with the
subcortical ones and takes the columns in order. Reported unadjusted, as a
Mann-Whitney U with Cliff's delta, and adjusted for diagnosis, mean framewise
displacement, age and sex.

### `abide_ho_s5_ctrl.py`

The same test for every model in the TransBrain table on both mappings, so that
every model is reported on the same footing, followed by the correlation of the
Magel2 score with head motion and with ADOS total within the case group.

### `abide_ho_s6_rot.py`, `abide_ho_s7_shank.py`

Rotation nulls for the Magel2 and Shank3 effects. Each draw rotates the mouse
coordinate frame, relabels each mouse parcel with the value of its nearest
rotated neighbour, routes the rotated pattern through π and repeats the
case-control test, so the spatial autocorrelation of the mouse pattern is held
fixed while its correspondence with anatomy is broken. Step 6 is resumable and
takes a wall-clock budget in seconds as its optional argument.

```bash
cd otter
python experiments/autism_subtypes/abide_subtype/abide_ho_s0_bundle.py
for s in s1_atlas s2_val s3_tpl s4_main s5_ctrl; do
    PYTHONPATH=src python experiments/autism_subtypes/abide_subtype/abide_ho_$s.py
done
PYTHONPATH=src python experiments/autism_subtypes/abide_subtype/abide_ho_s6_rot.py 600
PYTHONPATH=src python experiments/autism_subtypes/abide_subtype/abide_ho_s7_shank.py
```

Steps 4 to 7 execute `abide_ho_core.py` rather than importing it, so that
everything it defines is available to them by name.
