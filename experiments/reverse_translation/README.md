# Reverse translation

These analyses apply the row-normalised OTTER coupling to human brain maps and
rank mouse parcels or structures as candidate experimental targets. Run commands
from the repository root with `PYTHONPATH=src`.

## Clinical disease dimensions

The disease-dimension analyses distinguish discovery, external confirmation
and independent validation.

| Analysis | Script | Machine-readable result | Interpretation |
|---|---|---|---|
| Alzheimer phenotype discovery | `09_ad_phenotype_dissociation.py` | `outputs/logs/reverse_translation_ad_phenotypes.json` | Frozen phenotype-to-target assignments tested in tau-PET and VBM maps |
| Alzheimer external confirmation | `10_ad_external_leads_confirmation.py` | `outputs/logs/reverse_translation_ad_leads_confirmation.json` | Frozen assignments tested in multisite LEADS tau-PET maps |
| TMS symptom circuits | `07_symptom_dissociation.py` | `outputs/logs/reverse_translation_symptom_dissociation.json` | Clinically supported human circuits translated to distinct mouse prescriptions |
| Parkinson stage | `08_pd_stage_progression.py` | `outputs/logs/reverse_translation_pd_stage_progression.json` | ENIGMA stage-dependent motor-to-interoceptive translation |
| Parkinson stage validation | `qpn_pd_stage_validation/` | `outputs/logs/reverse_translation_qpn_pd_stage*.json` | Restricted QPN-NC participant-level and surface-resolution validation |

The corresponding overview is [`notebooks/09_disease_dimensions.ipynb`](../../notebooks/09_disease_dimensions.ipynb).
The notebook reads the committed JSON results by default and exposes an opt-in
cell for rerunning the producing scripts.

### Reproduction order

```bash
PYTHONPATH=src python experiments/reverse_translation/09_ad_phenotype_dissociation.py
PYTHONPATH=src python experiments/reverse_translation/10_ad_external_leads_confirmation.py
PYTHONPATH=src python experiments/reverse_translation/07_symptom_dissociation.py
PYTHONPATH=src python experiments/reverse_translation/08_pd_stage_progression.py
PYTHONPATH=src python experiments/reverse_translation/qpn_pd_stage_validation/qpn_stage_validation.py --archive-dir ../data_external/qpn-nc-r01
PYTHONPATH=src python experiments/reverse_translation/qpn_pd_stage_validation/qpn_surface_validation.py --archive-dir ../data_external/qpn-nc-r01
```

The TMS analysis uses 10,000 joint bilateral-pair Moran spectral
randomisations and may take several minutes. The Alzheimer and Parkinson
analyses use 10,000 mirrored, one-to-one spatial rotations. Every released log
records the null definition and random seed. The QPN-NC logs omit
restricted-file hashes and participant-level values.

## Evidence boundaries

- The Alzheimer discovery maps use fixed phenotype-to-target assignments. The
  same assignments pass the prespecified LEADS confirmation gate under both
  canonical OTTER and the relevant-regional-packs-removed refit. Exact
  participant non-overlap between the discovery and LEADS aggregates cannot be
  established, so the confirmation is described as largely independent.
- The TMS result is confirmatory for separation of the translated mouse targets
  under the primary parcel-mass statistic and survives removal of the directly
  relevant regional packs. Prospective clinical evidence validates the human
  input circuits, not the translated mouse targets. The connectivity increment
  does not pass every weighting sensitivity.
- The ENIGMA Parkinson stage pattern is supported in the independent QPN-NC
  cohort using control-normative cortical thinning, participant-level stage
  scores and a native FreeSurfer surface-resolution analysis. All analyses are
  cross-sectional and do not establish within-person progression.

## Source data

- The TMS atlas and download provenance are in
  [`clinical_maps/DATA_SOURCES.md`](clinical_maps/DATA_SOURCES.md).
- Alzheimer discovery-map provenance is in
  [`ad_phenotype_maps/DATA_SOURCES.md`](ad_phenotype_maps/DATA_SOURCES.md).
- Parkinson stage-map provenance and expected local filenames are in
  [`pd_stage_maps/DATA_SOURCES.md`](pd_stage_maps/DATA_SOURCES.md).
- The restricted QPN-NC input layout and privacy boundary are documented in
  [`qpn_pd_stage_validation/README.md`](qpn_pd_stage_validation/README.md).
- LEADS maps are downloaded on demand from NeuroVault collection 23001 by the
  confirmation script and stored under ignored `data_external/`.

Earlier scripts `01_validate.py` through `06_neuromaps_substrate.py` provide the
known-system and molecular positive controls that precede the clinical
disease-dimension analyses. `06_neuromaps_substrate.py` records individual
dopamine, serotonin, μ-opioid, CB1 and GABA-A striatal mass fractions.
`11_reverse_direction_diagnostic.py` mirrors the canonical semirelaxed solve
and records direction-specific coverage and agreement diagnostics.
