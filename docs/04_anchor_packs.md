# Anchor Packs

A **pack** is a small, self-contained Python module that builds one or more cross-species region anchors from published anatomical literature. Each entry pairs a set of mouse parcels with a set of human parcels and is applied to the FGW cost matrix as a *soft* constraint (default `lam_outside=0.15`, see [02_methods.md](02_methods.md)).

Packs are modular. Compose any subset, drop any subset. Default packs are layered together into the canonical coupling `outputs/coupling/pi_canonical.npy`, which also applies the anchor-warped spatial cost. The recipe below omits that warp and reproduces the retired pre-warp coupling; use `load_pi()` for the canonical one.

## Philosophy

Each pack reflects a **single published cross-species correspondence**. The mouse-side set comes from the DSURQE atlas overlay; the human-side set comes from canonical MNI cytoarchitectural centroids (Mai/Paxinos, Glasser HCP-MMP360, or atlas-specific references).

This is *not* a generative method, packs encode what's already known about anatomy, not what HOMER discovered. The FGW solver then propagates the constraints through FC + SC structure to fill in the unanchored ~80 % of parcels.

## Pid registry

All citations have been verified against the literature (Consensus search, May 2026). Citation count + journal links provided per pack data sheet below.

The **"In recommended π?"** column is authoritative against `src/homer/data/anchor_packs/registry.py`, the single source of truth for which packs are composed into the canonical coupling. **All 15 packs are in the recommended composition** (26 region-anchor entries): a multi-benchmark comparison showed the full set wins the TransBrain region-level homology benchmark decisively and ties for best on Beauchamp. A few packs carry a Beauchamp-metric trade-off (flagged in their data sheets below); they are kept because the broader evidence favours inclusion.

| pid range | Pack | In recommended π? | Primary reference |
|---|---|---|---|
| 1..21 | Garin point anchors (one mouse parcel ↔ one human parcel per pair) | yes | Garin 2021 |
| 30, 31 | BICCN motor (M1, M2 / PMd) | yes | [Bakken 2021, *Nature*](https://consensus.app/papers/details/82aefc336c0e5f2e88d65f51d91cfbfe/) |
| 32, 33 | Tectum (Superior + Inferior Colliculus) | yes | [Isa 2021, *Curr Biol*](https://consensus.app/papers/details/b167c990210e55e7923df8ebdf731a32/); Winer & Schreiner 2005 |
| 34, 35 | Olfactory (Piriform + Anterior olfactory nucleus) | yes | [Mori 2014](https://consensus.app/papers/details/0db38e2b1d39564799a5f173c4d942b1/); [Carlén 2017, *Science*](https://consensus.app/papers/details/8f13a81410c4529c920311c591ff7833/) |
| 36, 37 | Cingulate (Subgenual ACC + Retrosplenial) | yes (trade-off) | [Vogt et al. 2012, *Brain Struct Funct*](https://consensus.app/papers/details/ad69e350c1925154a579cd8ab2259311/) |
| 38 | Amygdala (Cortical subplate) | yes | [Janak & Tye 2015, *Nature*](https://consensus.app/papers/details/51a6d86145eb5376a83388d1d98475eb/); [Pessoa & Adolphs 2010, *Nat Rev Neurosci*](https://consensus.app/papers/details/e2ba7247dbca506a85f4d75eaf008c49/) |
| 39-42 | Hippocampal (Subi + CA1 + CA3 + Dentate gyrus) | yes | [Strange et al. 2014, *Nat Rev Neurosci*](https://consensus.app/papers/details/825d36a33ecd562c9c8a572b9930dd51/); [Iglesias et al. 2015, *NeuroImage*](https://consensus.app/papers/details/c7d5cd3753935868968799a20b664da7/) |
| 45 (46 opt-in) | Lateral PFC. OFC default; dlPFC contested, excluded | yes (OFC only) | [Wallis 2011, *Nat Neurosci*](https://consensus.app/papers/details/580020c7d32e54d4bb2b4f3270b6a2b2/); [Carlén 2017, *Science*](https://consensus.app/papers/details/8f13a81410c4529c920311c591ff7833/) vs [Preuss 1995, *J Cogn Neurosci*](https://consensus.app/papers/details/2e000a3af07f508489ac7ba2f68c68dc/) |
| 47, 48 | Striatum (CP dorsolateral, CP ventromedial) | yes | [Voorn et al. 2004, *TINS*](https://consensus.app/papers/details/abd59449cc065adfa2988e4c7511869c/) |
| 49 | Entorhinal cortex | yes | [Franjic et al. 2021, *Neuron*](https://consensus.app/papers/details/8133c8accfab51e6892690a7bed0c27e/) |
| 52 | Visual extrastriate (mouse LM ↔ human V2) | yes (trade-off) | [Wang & Burkhalter 2007, *J Comp Neurol*](https://consensus.app/papers/details/b4b515e8765d5045976cb27b170a865b/) |
| 54 | Periaqueductal gray (PAG) | yes | [Ezra et al. 2015, *Hum Brain Mapp*](https://consensus.app/papers/details/38ef392c65c5502d8004b70a100c0d55/); [Kingsbury et al. 2011, *PLOS ONE*](https://consensus.app/papers/details/a1924b15fe8f5245bdbe097e02235646/) |
| 55 | Perirhinal cortex | yes | [Burwell et al. 1995, *Hippocampus*](https://consensus.app/papers/details/c76cb6a4230b5090b0c27d6b1685c7aa/) |
| 56, 57 | Auditory core + belt (A1 + A2-dorsal/ventral) | yes | [Hackett et al. 2001, *J Comp Neurol*](https://consensus.app/papers/details/052fa94346785c9b8a06d58cbe6f651d/); [Kaas & Hackett 2000, *PNAS*](https://consensus.app/papers/details/275d7ce7c42857c7b78dc8f2a9fb3b16/) |
| 58, 59, 60 | Somatosensory body-map (face, hand, leg) | yes (trade-off) | Penfield 1937; [Seelke et al. 2012, *PLOS ONE*](https://consensus.app/papers/details/668dd505c7205f73ae24bd50fdf577c0/); [Gordon et al. 2023, *Nature*](https://consensus.app/papers/details/f24dace36aa05982a170fdc5de32b051/) |
| 61 | Posterior parietal cortex (BA7) | yes | [Whitlock 2017, *Current Biology*](https://consensus.app/papers/details/27e90f12f5b95e2c8f6c8e0a25c33c5d/) |
| ≥ 62 | (reserved for future packs, see Roadmap below) | | |

## Pack data sheets

### biccn_motor, pids 30, 31

- **Source**: [Bakken et al. 2021, *Nature*](https://consensus.app/papers/details/82aefc336c0e5f2e88d65f51d91cfbfe/). BICCN Motor Cortex Consortium cell-type taxonomy (546 citations). DOI: 10.1038/s41586-021-03465-8.
- **pid 30**: Mouse Primary motor area (53 parcels via DSURQE) ↔ Human BA4 at MNI(±37, −22, 55) r=10 mm (12 parcels)
- **pid 31**: Mouse Secondary motor area (48 parcels) ↔ Human PMd / Area 6 at MNI(±28, −5, 62) r=12 mm (23 parcels)
- **Lifts**: Beauchamp "Primary motor area → precentral gyrus" 0 → 100 %
- **Off-target**: S1 top-5 −2 pp
- **Held-out test**: Fitting with M2 anchor only leaves M1 at 0 % top-1, structure doesn't recover the held-out anchor.

### tectum, pids 32, 33

- **Sources**:
  - [Isa et al. 2021, *Current Biology*](https://consensus.app/papers/details/b167c990210e55e7923df8ebdf731a32/), "The tectum/superior colliculus as the vertebrate solution for spatial sensory integration and action" (139 cit).
  - Winer & Schreiner 2005, "The inferior colliculus" (Springer book; the canonical IC reference).
  - Both establish conserved SC + IC cytoarchitecture across vertebrates.
- **pid 32**: Mouse Superior Colliculus sensory (53 parcels) ↔ Human SC at MNI(±5, −30, −2) r=6 mm (2 parcels)
- **pid 33**: Mouse Inferior Colliculus (29 parcels) ↔ Human IC at MNI(±5, −35, −8) r=8 mm (4 parcels)
- **Lifts**: Beauchamp SC + IC 0 → 100 %
- **Off-target**: Thalamus −4 pp
- **Held-out**: SC anchor alone leaves IC at 0 %.

### olfactory, pids 34, 35

- **Sources**:
  - [Mori 2014](https://consensus.app/papers/details/0db38e2b1d39564799a5f173c4d942b1/), "The Olfactory System" (Springer book chapter; 260 cit).
  - [Carlén 2017, *Science*](https://consensus.app/papers/details/8f13a81410c4529c920311c591ff7833/), "What constitutes the prefrontal cortex?" (432 cit; includes olfactory PFC connectivity homologies).
- **pid 34**: Mouse Piriform area (47 parcels) ↔ Human Piriform cortex at MNI(±25, 5, −20) r=10 mm (13 parcels)
- **pid 35**: Mouse Anterior olfactory nucleus (9 parcels) ↔ Human AON at MNI(±15, 25, −15) r=10 mm (6 parcels)
- **Lifts**: Beauchamp Piriform 0 → 100 %
- **Off-target**: None, cleanest pack
- **Composition caveat**: shares 2 human parcels (L/R Olfactory cortex) with the amygdala pack, see amygdala below.

### cingulate, pids 36, 37 (in recommended π; Beauchamp trade-off)

- **Sources**:
  - [Vogt et al. 2012, *Brain Structure and Function*](https://consensus.app/papers/details/ad69e350c1925154a579cd8ab2259311/), "Cytoarchitecture of mouse and rat cingulate cortex with human homologies" (318 cit). DOI: 10.1007/s00429-012-0411-8. **Primary reference for the area-32 subgenual/pregenual subdivisions used here.**
  - Vogt et al. 2013, *J Comp Neurol*, "Cingulate area 32 homologies in mouse, rat, macaque and human" (94 cit). Companion paper.
  - van Heukelum et al. 2020, *Trends in Neurosciences*, "Where is Cingulate Cortex? A Cross-Species View" (213 cit). Recent reframing.
- **pid 36**: Mouse ACA ventral (15 parcels) ↔ Human subgenual ACC at MNI(±5, 10, 35) r=10 mm (6 parcels)
- **pid 37**: Mouse Retrosplenial (27 parcels) ↔ Human RSC at MNI(±15, −55, 10) r=10 mm (8 parcels)
- **Beauchamp ACG impact**: 13 % → **9 %** (anchor target = subgenual ACC, validation target = pregenual ACC). Documented hurt.
- **Off-target**: None (besides ACG)
- **Beauchamp trade-off**: the metric drop on ACG is real and reflects a real anatomical disagreement between anchor location and validation location. The pack stays in the recommended composition, it is anatomically defensible and the broad multi-benchmark evidence favours inclusion.

### amygdala, pid 38

- **Sources**:
  - [Janak & Tye 2015, *Nature*](https://consensus.app/papers/details/51a6d86145eb5376a83388d1d98475eb/), "From circuits to behaviour in the amygdala" (1701 cit). DOI: 10.1038/nature14188.
  - [Pessoa & Adolphs 2010, *Nature Reviews Neuroscience*](https://consensus.app/papers/details/e2ba7247dbca506a85f4d75eaf008c49/), "Emotion processing and the amygdala, from a 'low road' to 'many roads'" (1663 cit). DOI 10.1038/nrn2920.
- **pid 38**: Mouse Cortical subplate (54 parcels) ↔ Human amygdala at MNI(±25, −5, −20) r=8 mm (6 parcels)
- **Single-entry pack**. DSURQE doesn't distinguish basolateral/central/lateral sub-nuclei
- **Lifts**: Beauchamp Cortical subplate → amygdala 0 → 100 %
- **Off-target**: None
- **Composition caveat**: shares 2 parcels (L/R Olfactory cortex) with olfactory pid 34. The FGW solver handles the conflict (soft constraints); mass on those 2 parcels ends up intermediate.

### hippocampal, pids 39, 40, 41, 42

- **Sources**:
  - [Strange et al. 2014, *Nature Reviews Neuroscience*](https://consensus.app/papers/details/825d36a33ecd562c9c8a572b9930dd51/), "Functional organization of the hippocampal longitudinal axis" (1503 cit). DOI: 10.1038/nrn3785.
  - [Iglesias et al. 2015, *NeuroImage*](https://consensus.app/papers/details/c7d5cd3753935868968799a20b664da7/), "A computational atlas of the hippocampal formation using ex vivo, ultra-high resolution MRI" (1139 cit). DOI: 10.1016/j.neuroimage.2015.04.042. Provides the MNI centroids for subfield balls.
- **pid 39**: Subiculum (29 mouse / 8 human)
- **pid 40**: CA1 (15 / 6)
- **pid 41**: CA3 (26 / 4)
- **pid 42**: Dentate gyrus (22 / 4)
- **Lifts**: All 4 subfields 0 → 100 % top-1
- **Off-target**: Thalamus +1 pp ripple, otherwise stable
- **CA2 skipped**: not in DSURQE tree
- **Held-out**: Anchoring Subiculum alone leaves CA1/CA3/DG at 0 %, confirms structure does not propagate across subfields.

### lateral_pfc, pid 45 OFC (in recommended π); pid 46 dlPFC (opt-in, excluded by default)

- **Sources**:
  - [Wallis 2011, *Nature Neuroscience*](https://consensus.app/papers/details/580020c7d32e54d4bb2b4f3270b6a2b2/), "Cross-species studies of orbitofrontal cortex and value-based decision-making" (347 cit). DOI: 10.1038/nn.2956. **OFC homology, high confidence.**
  - [Carlén 2017, *Science*](https://consensus.app/papers/details/8f13a81410c4529c920311c591ff7833/), "What constitutes the prefrontal cortex?" (432 cit). DOI: 10.1126/science.aan8868. Argues functional PFC homology.
  - [Preuss 1995, *J Cognitive Neuroscience*](https://consensus.app/papers/details/2e000a3af07f508489ac7ba2f68c68dc/), "Do Rats Have Prefrontal Cortex? The Rose-Woolsey-Akert Program Reconsidered" (684 cit). DOI: 10.1162/jocn.1995.7.1.1. **Argues against rodent dlPFC homology.**
  - Laubach 2018, *eNeuro*, "What, If Anything, Is Rodent Prefrontal Cortex?" (351 cit). Modern continuation of debate.
- **pid 45. OFC**: Mouse Orbital area lateral (21 parcels) ↔ Human OFC BA11/47 at MNI(±25, 35, −15) r=10 mm (8 parcels). High confidence.
- **pid 46, dlPFC (contested, opt-in)**: Mouse Prelimbic (11 parcels) ↔ Human dlPFC BA9/46 at MNI(±40, 25, 35) r=10 mm (12 parcels). **Preuss 1995 argues rodents lack a true dlPFC; Carlén 2017 / Laubach 2018 argue functional homology.** This entry is **excluded from the recommended composition by default**, the homology is contested and HOMER's own Schaeffer et al. 2020 falsification test contradicts it (forcing the anchor routes 23 % of mouse-MFC mass to dlPFC by construction). Pass `build_lateral_pfc_region_anchors(..., include_dlpfc=True)` to add it for ablations.
- **Lifts**: Neither OFC nor dlPFC has a Beauchamp validation pair, purely anatomical-credibility supervision.
- **Off-target**: None measurable.

### striatum, pids 47, 48

- **Source**: [Voorn et al. 2004, *Trends in Neurosciences*](https://consensus.app/papers/details/abd59449cc065adfa2988e4c7511869c/), "Putting a spin on the dorsal-ventral divide of the striatum" (1198 citations). DOI: 10.1016/j.tins.2004.06.006. Reframes the classic dorsal-ventral striatum divide as a **mediolateral functional gradient**.
- **pid 47. Dorsolateral CP**: Mouse Caudoputamen subset with |x| > median and z > median (~18 parcels) ↔ Human putamen at MNI(±28, 0, 0) r=10 mm (~18 parcels). Sensorimotor striatum.
- **pid 48. Ventromedial CP**: Mouse Caudoputamen subset with |x| ≤ median and z ≤ median (~46 parcels) ↔ Human caudate anterior at MNI(±10, 10, 10) r=10 mm (~8 parcels). Limbic/associative striatum.
- **Lifts**: Beauchamp **"Caudoputamen → caudate nucleus" 13 % → 33 %** top-1 (+19 pp); NAc 8 % → 12 % (+4 pp); ACG 13 % → 17 % (+4 pp).
- **Off-target**: None detected.
- **Note on subset selection**: DSURQE doesn't expose dorsolateral / ventromedial as separate labels, so we partition Caudoputamen parcels by (|x|, z) coordinates relative to their median. The ~85 "middle gradient" parcels are not in either subset. Voorn's gradient view explicitly says there's no sharp boundary, and Beauchamp's existing Caudoputamen target + Garin pid 13 cover those.

### entorhinal, pid 49

- **Source**: [Franjic et al. 2021, *Neuron*](https://consensus.app/papers/details/8133c8accfab51e6892690a7bed0c27e/), "Transcriptomic taxonomy and neurogenic trajectories of adult human, macaque, and pig hippocampal and entorhinal cells" (222 citations). DOI: 10.1016/j.neuron.2021.10.036. Direct cross-species cell-type homology.
- **pid 49**: Mouse Entorhinal area (84 parcels) ↔ Human entorhinal cortex at MNI(±20, −10, −30) r=10 mm (6 parcels).
- **Lifts**: Beauchamp doesn't have an entorhinal validation pair, purely anatomical-credibility supervision. Hippocampal Subiculum top-1 unaffected (entorhinal sits separately).
- **Off-target**: None detected.
- **Single-entry pack**. DSURQE exposes "Entorhinal area" and "Entorhinal area, lateral part" but no "medial part" label, so we can't yet split into lateral-EC (object/contextual memory) vs medial-EC (spatial/grid-cell) per Ohara 2021. pid 50 reserved for a future split if DSURQE adds the medial-part label.

### visual, pid 52

- **Source**: [Wang & Burkhalter 2007, *J Comp Neurol*](https://consensus.app/papers/details/b4b515e8765d5045976cb27b170a865b/), "Area map of mouse visual cortex" (481 citations). DOI: 10.1002/cne.21286. Established the **mouse LM ↔ primate V2** homology via retinotopy + V1 input pattern + laminar architecture.
- **pid 52**: Mouse Lateral visual area (LM, 9 parcels) ↔ Human V2 at MNI(±20, −85, 10) r=10 mm (12 parcels).
- **Lifts**: Beauchamp "Visual areas → cuneus" stays at 7 % (Beauchamp's validation uses all 54 mouse Visual parcels, not the 9 LM subset, and cuneus ball at (±10, -85, 5) rather than our V2 at (±20, -85, 10), the difference doesn't surface in the Beauchamp metric).
- **Off-target**: None detected.
- **Value**: anatomical-credibility supervision; makes the specific LM↔V2 query trustworthy for downstream users studying visual hierarchy. AL↔V3 and AM↔V4 mappings are more contested and not included; pid 53 reserved for a future split.

### pag, pid 54

- **Sources**:
  - [Ezra et al. 2015, *Human Brain Mapping*](https://consensus.app/papers/details/38ef392c65c5502d8004b70a100c0d55/), "Connectivity-based segmentation of the periaqueductal gray matter in human" (76 cit). DOI: 10.1002/hbm.22855. Identifies four PAG sub-columns in humans concordant with rodent model.
  - [Kingsbury et al. 2011, *PLOS ONE*](https://consensus.app/papers/details/a1924b15fe8f5245bdbe097e02235646/). Mammal-like PAG columnar organization in birds (93 cit). Pan-amniote conservation.
- **pid 54**: Mouse Periaqueductal gray (16 parcels) ↔ Human PAG at MNI(±5, −30, −10) r=6 mm (4 parcels).
- **Lifts**: No PAG validation pair in Beauchamp. Side effect: NAc rises +4 pp top-1 (8 % → 12 %), possibly via midbrain→forebrain mass redistribution.
- **Off-target**: None negative.
- **Caveat from Ezra 2015**: PAG *columnar structure* is conserved across species, but *cortical connectivity* differs. The gross PAG↔PAG anchor is defensible; sub-column splits (dorsolateral / lateral / ventrolateral) are NOT attempted because the human PAG sub-column atlas is research-grade and not standardised.

### perirhinal, pid 55

- **Sources**:
  - [Burwell et al. 1995, *Hippocampus*](https://consensus.app/papers/details/c76cb6a4230b5090b0c27d6b1685c7aa/), canonical rat-monkey perirhinal/postrhinal anatomy review (544 citations). DOI: 10.1002/hipo.450050503. Established the rodent perirhinal–postrhinal nomenclature and primate homology.
  - Kealy & Commins 2011, *Progress in Neurobiology*, rat perirhinal anatomy/physiology review (123 cit). DOI: 10.1016/j.pneurobio.2011.03.002.
- **pid 55**: Mouse Perirhinal area (6 parcels) ↔ Human perirhinal cortex at MNI(±35, −10, −30) r=10 mm (6 parcels).
- **Lifts**: No perirhinal validation pair in Beauchamp. Completes HOMER's MTL coverage (hippocampal + entorhinal + perirhinal).
- **Off-target**: None detected.
- **Value**: anatomical-credibility supervision for memory researchers studying object-recognition / familiarity memory.

### auditory, pids 56, 57

- **Sources**:
  - [Hackett et al. 2001, *J Comp Neurol*](https://consensus.app/papers/details/052fa94346785c9b8a06d58cbe6f651d/), "Architectonic identification of the core region in auditory cortex of macaques, chimpanzees, and humans" (464 citations). DOI: 10.1002/cne.1407.
  - [Kaas & Hackett 2000, *PNAS*](https://consensus.app/papers/details/275d7ce7c42857c7b78dc8f2a9fb3b16/), "Subdivisions of auditory cortex and processing streams in primates" (1025 citations). DOI: 10.1073/pnas.97.22.11793.
- **pid 56. A1 core**: Mouse Primary auditory area (9 parcels) ↔ Human A1 core (BA41) at MNI(±48, −22, 6) r=6 mm (2 parcels).
- **pid 57. Auditory belt**: Mouse Dorsal+Ventral auditory areas (11 parcels combined) ↔ Human auditory belt (BA42) at MNI(±55, −15, 0) r=8 mm (4 parcels).
- **Lifts**: Beauchamp **"Primary auditory area → Heschl's gyrus" 22 % → 100 %** top-1 (+78 pp). The largest single-pack lift on a previously-anchored region.
- **Off-target**: None detected.
- **Why this works**: A1 anchor's mouse-side set is identical to Beauchamp's validation set, and the A1-core human ball sits *inside* Beauchamp's Heschl's gyrus ball, so the soft constraint concentrates mass on the canonical target without competing with other regions.

### somatosensory, pids 58, 59, 60 (in recommended π; Beauchamp trade-off)

- **Sources**:
  - Penfield 1937, canonical sensorimotor homunculus.
  - [Seelke et al. 2012, *PLOS ONE*](https://consensus.app/papers/details/668dd505c7205f73ae24bd50fdf577c0/), emergence of somatotopic body maps in rat S1 (77 cit). DOI: 10.1371/journal.pone.0032322.
  - [Gordon et al. 2023, *Nature*](https://consensus.app/papers/details/f24dace36aa05982a170fdc5de32b051/), somatosensory body map cross-species, including macaque/human homologues (345 cit). DOI: 10.1038/s41586-023-05964-2.
  - Freire et al. 2024, *Brain Behav Evol*, somatotopy in non-laboratory rodent confirming common mammalian plan.
- **pid 58. Face S1**: Mouse Barrel field + Nose (88 parcels combined) ↔ Human Face S1 BA3b ventral at MNI(±55, −15, 25) r=8 mm (6 parcels).
- **pid 59. Hand S1**: Mouse Upper limb (24 parcels) ↔ Human Hand S1 at MNI(±40, −25, 55) r=8 mm (4 parcels).
- **pid 60. Leg S1**: Mouse Lower limb (14 parcels) ↔ Human Leg S1 / paracentral at MNI(±10, −40, 70) r=10 mm (12 parcels).
- **Beauchamp impact**: "Primary somatosensory area → postcentral gyrus" **20 % → 15 %** (−5 pp). Same dynamic as the cingulate pack, where the anchor target is not the validation target. Beauchamp's r=15 ball is centred at hand S1 (-40, -25, 55); our face S1 ball at (-55, -15, 25) is ~35 mm away, outside the validation ball.
- **Off-target**: None outside S1.
- **Beauchamp trade-off**: it lowers the Beauchamp S1 *parcel* metric, but adds body-map-specific S1 structure (Penfield-style face/hand/leg distinction). Anatomical credibility is unimpeachable; Beauchamp's broad-ball validation just measures something coarser. The pack stays in the recommended composition.

### ppc, pid 61

- **Sources**:
  - [Whitlock 2017, *Current Biology*](https://consensus.app/papers/details/27e90f12f5b95e2c8f6c8e0a25c33c5d/), "Posterior parietal cortex" cross-species review.
  - Lyamzin & Benucci 2019, *Neuroscience Research*, mouse PPC review.
- **pid 61**: Mouse Posterior parietal association areas (10 parcels) ↔ Human PPC BA7 at MNI(±35, −55, 50) r=10 mm (14 parcels).
- **Lifts**: No PPC validation pair in Beauchamp (Garin pid 4 covers Posterior Parietal as a point anchor; this pack adds region coverage).
- **Off-target**: None detected.

## Composition recipe

The recommended π is composed from the `default=True` packs in the registry
(`src/homer/data/anchor_packs/registry.py`), currently all 15 packs
(26 region-anchor entries). Use `build_default_pack_entries` rather than
re-listing the builders by hand:

```python
import numpy as np
from homer.data import load_cached
from homer.data.anchor_packs import build_default_pack_entries
from homer.models import MultimodalFGW

M, _ = load_cached("mouse", cache_dir="outputs/anndata")
H, _ = load_cached("human", cache_dir="outputs/anndata")
costs = np.load("outputs/anndata/full_costs.npz")

entries = build_default_pack_entries(M.var, H.var)

model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                      epsilon=0.05, xyz_weight=0.25, lam_anchor=1.0)
model.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
          region_anchors=entries)
np.save("outputs/coupling/pi_fc_plus_SC_with_all_packs.npy", model.pi)
```

To add a pack to the recommended composition, flip its `default` flag in the
registry, every consumer (compose script, GUI, trust step) picks it up.

Run end-to-end via `experiments/anchor_packs/compose_all.py`, or the whole
recommended-model pipeline via `pipeline/run_recommended_model.py`. The
non-default packs above remain available, compose them explicitly (or run
`experiments/anchor_packs/<pack>.py`) for ablations and targeted queries.

## Tuning the soft constraint

Each region anchor uses `lam_outside=0.15` (soft) by default. To override:

```python
model.fit(..., region_anchors=entries, region_lam_outside=1.0)  # hard 0/1 wall
```

The soft default gives the same argmax as hard, with better-calibrated probability tails, see [archive/iteration_log.md §5.6.0a](archive/iteration_log.md).

## Working around the DSURQE granularity limit

**The problem**: DSURQE doesn't expose some well-conserved cross-species structures as labels (habenula, locus coeruleus, substantia nigra, VTA, claustrum, raphe nuclei). For each, we know the cross-species homology is real and published (Aizawa 2011 habenula; Manger 2021 LC; Krashia 2017 SN/VTA; Smith 2018 claustrum), but we can't get a mouse-side parcel set via DSURQE label lookup.

**The workaround**: ``homer.data.anchor_packs._dsurqe.mouse_parcels_in_mouse_sphere``, symmetric to the human-side helper. Pass a centroid in M_var coords + radius; get back the mouse parcels in that vicinity. Lets you build packs without going through DSURQE labels.

**What it does and doesn't deliver**:

- **Mechanically works** for medium-sized structures (claustrum, SN region) at r ≈ 0.5−1.0 mm.
- **Anatomical specificity is lost**, the captured parcels are in the right spatial neighbourhood but not labelled as the target structure. We're trusting coordinates rather than verified labels.
- **Very small structures** (habenula ~0.5 mm³, LC ~0.1 mm³) at our 200μm parcel resolution. Even tight balls (r=0.3-0.5) capture neighbouring thalamic/midbrain parcels rather than the structure proper.
- **No Beauchamp validation pair** for any of these structures, so empirical assessment isn't possible.

**Conclusion**: the workaround is real infrastructure but **not yet a recipe we trust enough to ship as default packs**. Building a Habenula or LC pack would require external verification (e.g. comparing captured parcels' gene expression against habenular/LC-specific markers from Allen ISH data, or against Yao 2023 cell-type composition). That's a larger curation effort, left as a deliberate future-work item.

## Roadmap, remaining candidate packs

The recommended composition is all 15 packs / 26 region-anchor entries (pids 30−61). Remaining candidates with strong literature support but not yet implemented:

| Candidate | Strong reference | Feasibility | Reserved pids |
|---|---|---|---|
| **Locus Coeruleus** | [Manger et al. 2021, *Brain Sciences*](https://consensus.app/papers/details/8b247b83a1fc566b9821c690df070906/) | DSURQE no label; needs `mouse_parcels_in_mouse_sphere` + verification | 51 |
| **Habenula** | Aizawa 2011, *Frontiers in Neuroscience*; Stephenson-Jones 2011, *PNAS* | DSURQE no label; coordinate-based with verification | 53 (between 52, 54) |
| **Substantia nigra / VTA** | Krashia 2017, *Eur J Neurosci*; Root 2016, *Sci Rep* | DSURQE no label; coordinate-based with verification | 55, 56 |
| **Claustrum** | [Smith et al. 2018, *J Comp Neurol*](https://consensus.app/papers/details/fdb04a6542615874a0b0383728b813ca/); Norimoto 2020, *Nature* | DSURQE label exists but 0 parcels (sub-resolution); coord-based viable | 57 |
| **Entorhinal lateral/medial split** | Ohara 2021; Franjic 2021 | DSURQE has lateral but not medial; derive medial via set difference | 50 |
| **Primate Insular subdivisions** | Evrard 2019 | DSURQE doesn't expose insular sub-divisions; **not feasible** at current parcellation | |
| **Whole-brain cell-type homology** | Yao 2023, *Nature*; Siletti 2023, *Science* | Different kind of pack, needs ABCA pipeline integration | (separate module) |

For brainstem/midbrain nuclei (LC, habenula, SN, VTA, claustrum), the verification step before shipping is to cross-check that the spatially-selected mouse parcels actually express markers of that structure (Allen ISH or Yao 2023 cell-type composition). Without that, we'd be shipping packs whose anatomical specificity is asserted but not verified.

## Adding a new pack

See [06_extending.md](06_extending.md) for the recipe. Pattern: pick a pid range ≥ 47 (so it doesn't clash with existing packs), drop a new module into `src/homer/data/anchor_packs/`, expose `build_<name>_region_anchors(M_var, H_var)`, add an integration test.
