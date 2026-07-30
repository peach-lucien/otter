# Disease reverse-translation — human disease map → mouse-model substrate (preregistered)

Each human disease is paired *a priori* with the mouse brain structure(s) where the field
actually builds and studies its models — decided before scoring, from the canonical
mouse-model literature. A hit = the reverse-translated human disease map ranks one of the
listed mouse structures in the top-3 (above a 1,000-rotation spin null). This is the same
pipeline validated on 12 functional systems (script 01); here the ground truth is the
mouse-model substrate rather than the normal-function substrate.

## Neurological diseases — strong, uncontroversial mouse-model substrate (expect HIT)

| Disease | Neurosynth term | Mouse-model substrate | Basis (mouse model) | Reference |
|---|---|---|---|---|
| Parkinson's | parkinson | SNc, SNr, CP, STN, VTA | dopaminergic nigrostriatal loss; 6-OHDA, MPTP, α-synuclein | Blesa & Przedborski 2014, *Front Neuroanat*; Dawson et al. 2010, *Neuron* |
| Alzheimer's | alzheimer | ENTl, ENTm, CA1, SUB, DG, CA3 | entorhinal–hippocampal tau/Aβ (Braak I–II); APP/PS1, tau lines | Braak & Braak 1991, *Acta Neuropathol*; Spires-Jones & Hyman 2014, *Neuron* |
| Huntington's | huntington | CP, ACB | striatal medium-spiny-neuron loss; R6/2, zQ175, YAC128 | Bates et al. 2015, *Nat Rev Dis Primers* |
| Temporal-lobe epilepsy | epilepsy | CA1, CA3, DG, ENTl, ENTm | hippocampal sclerosis; kainate, pilocarpine | Lévesque & Avoli 2013, *Neurosci Biobehav Rev* |
| Addiction | addiction | ACB, VTA | mesolimbic dopamine; self-administration | Lüscher 2016, *Annu Rev Neurosci* (10.1146/annurev-neuro-070815-013920) |

## Psychiatric conditions — contested / poor mouse-model substrate (expect MISS or distributed)

| Condition | Neurosynth term | Mouse substrate often invoked | Note |
|---|---|---|---|
| Schizophrenia | schizophrenia | CA1, SUB, PL, ILA, CP | contested; hippocampal hyperactivity (MAM, DISC1), mPFC, striatal DA |
| OCD | obsessive | CP, ACB | cortico-striatal; SAPAP3, Slitrk5 striatal models (moderate) |
| Autism | autism | — (distributed) | deliberately expected NO clean mouse home — the primate-elaborated / poor-model case |

## Why this is a fair, preregistered test
- The mouse substrates are set from the *model* literature, not from HOMER's output, and the
  neurological pairings (nigrostriatal, entorhinal–hippocampal, striatal, mesolimbic) are
  textbook-uncontroversial.
- Autism is the built-in expected-negative (as addiction was in the function table): a
  distributed, human-elaborated condition without a single mouse home; failure to clear the
  null is the "no adequate mouse model" signal, not a bug.
- DOIs for the disease-model reviews should be PubMed-verified before publication; the
  substrate pairings themselves are not in dispute.
