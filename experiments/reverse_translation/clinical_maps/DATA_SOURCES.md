# Clinical network maps — exact download links

Robust, distributed whole-brain MNI152 voxel network maps (not point seeds), so they route
through the coupling the same way the Neurosynth functional maps did. All four below are
directly downloadable from **NeuroVault collection 13075** ("Causal mapping of human brain
function", Siddiqi lab): https://neurovault.org/collections/13075/

| stem (rename to) | map | n | reference (DOI, PubMed-verified) | direct file URL |
|---|---|---|---|---|
| `depression_tms` | convergent depression TMS circuit | 713 | Siddiqi et al. 2021, *Nat Hum Behav* — 10.1038/s41562-021-01161-1 | https://neurovault.org/media/images/13075/DepressionCircuit_t.nii.gz |
| `tms_anxdys` | dysphoric/anxiosomatic TMS targeting atlas (+=anxiosomatic, −=dysphoric) | 111 | Siddiqi et al. 2020, *Am J Psychiatry* — 10.1176/appi.ajp.2019.19090915 | https://neurovault.org/media/images/13075/TargetAtlas_AnxDys.nii.gz |
| `ptsd_circuit` | PTSD circuit | 193 | Siddiqi et al. 2024, *Nat Neurosci* — 10.1038/s41593-024-01772-7 | https://neurovault.org/media/images/13075/PTSDmap.nii.gz |
| `ms_depression` | MS-depression circuit (age/sex/EDSS/vol-corrected) | 281 | Siddiqi et al. 2023, *Nat Mental Health* — 10.1038/s44220-022-00002-y | https://neurovault.org/media/images/13075/DepMap_cf_AgeSexEdssVol.nii.gz |

## One-shot download (run from this folder)

```bash
cd homer/experiments/reverse_translation/clinical_maps
curl -L -o depression_tms.nii.gz  https://neurovault.org/media/images/13075/DepressionCircuit_t.nii.gz
curl -L -o tms_anxdys.nii.gz      https://neurovault.org/media/images/13075/TargetAtlas_AnxDys.nii.gz
curl -L -o ptsd_circuit.nii.gz    https://neurovault.org/media/images/13075/PTSDmap.nii.gz
curl -L -o ms_depression.nii.gz   https://neurovault.org/media/images/13075/DepMap_cf_AgeSexEdssVol.nii.gz
```

The script `03_clinical_networks.py` reads whatever of these are present and skips the rest.
NeuroVault persistent citation IDs (put in the manuscript if you use a map), e.g. the
anxdys atlas: `https://identifiers.org/neurovault.image:787858`.

## Subcortical DBS optimal targets (OCD / PD) — extra conversion needed
The Li/Baldermann OCD-DBS and Horn PD-STN-DBS *optimal targets* are published as **fiber-tract
atlases** (streamlines) distributed inside Lead-DBS (https://www.lead-dbs.org), not as MNI voxel
NIfTIs. To feed them here they must first be rendered to an MNI voxel volume (e.g. tract density
/ connectivity R-map). Papers:
- OCD unified connectomic target — Li, Baldermann, Horn et al. 2020, *Nat Commun* — 10.1038/s41467-020-16734-3
- OCD response tract (Biol Psychiatry) — Baldermann/Li/Horn 2021 — 10.1016/j.biopsych.2021.07.010
- PD STN-DBS connectivity — Horn et al. 2017, *Ann Neurol* — 10.1002/ana.24974
The conserved-subcortical routing they would test is already demonstrated by `01_validate.py`
(reward→ACB/VTA, fear→amygdala, etc.; 12/12 spin-significant), so these are optional.
