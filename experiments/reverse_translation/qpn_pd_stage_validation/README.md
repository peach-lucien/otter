# QPN-NC Parkinson stage validation

This analysis tests whether cortical thinning associated with greater Parkinson
disease severity is preferentially translated by OTTER towards mouse visceral,
gustatory and agranular-insular cortex rather than primary motor cortex.

The analysis uses two spatial resolutions from the restricted QPN-NC release:

- 50 Desikan--Killiany cortical regions shared with the ENIGMA stage maps;
- 148 bilateral Destrieux regions obtained from the native FreeSurfer outputs.

At both resolutions, cortical thinning is expressed relative to an age- and
sex-adjusted control model. Regional stage effects adjust for MRI age, sex and
the interval between MRI and clinical assessment. The directional contrast is
the mean translated weight for `VISC`, `GU`, `AId`, `AIv` and `AIp` minus the
mean translated weight for `MOp` and `MOs`. Spatial inference uses 10,000
hemisphere-preserving, one-to-one rotations. The coarse-resolution analysis
also evaluates a participant-level partial Spearman association, robust and
alternative stage formulations, a synchronized maximum-statistic test, and
leave-one-participant-out stability.

## Restricted input data

QPN-NC is a restricted-access dataset. Request access from the data owners at
[Zenodo record 17246063](https://doi.org/10.5281/zenodo.17246063). Place the
approved downloads outside this Git repository, for example:

```text
../data_external/qpn-nc-r01/
├── structural_measures.tar
├── tabular.tar
└── freesurfer_v7.3.2/
    ├── freesurfer_v7.3.2_001-013.tar
    └── ... 20 additional archives
```

Do not copy QPN-NC archives, extracted tables, participant-level derivatives or
identifiers into this repository. The scripts read the archives in place and
write only aggregate results. The committed result files contain no participant
identifiers, participant maps or participant scores.

The public Zenodo page states that access is granted case by case and that the
data usage agreement accompanies the custom download link. Users remain
responsible for complying with the agreement issued with their own access.

## Run

From the repository root:

```bash
PYTHONPATH=src python experiments/reverse_translation/qpn_pd_stage_validation/qpn_stage_validation.py \
  --archive-dir ../data_external/qpn-nc-r01

PYTHONPATH=src python experiments/reverse_translation/qpn_pd_stage_validation/qpn_surface_validation.py \
  --archive-dir ../data_external/qpn-nc-r01
```

The default outputs are:

- `outputs/logs/reverse_translation_qpn_pd_stage.json`
- `outputs/logs/reverse_translation_qpn_pd_stage_surface.json`

Both are aggregate summaries. They intentionally omit participant-level values
and restricted-file manifests.
