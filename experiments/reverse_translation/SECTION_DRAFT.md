# Draft results section — Reverse translation (for review before folding into the manuscript)

Style follows the manuscript: declarative, name the measure and what is compared, no em-dashes,
no AI tics. Numbers are from the locked runs; [ref] markers flag citations to add (DOIs to
verify via PubMed before submission).

---

## Reverse translation prescribes the mouse substrate of a human target

The coupling is bidirectional. Column-normalised it carries mouse signals to human cortex
(previous section); row-normalised it carries a human map to mouse, assigning each mouse region
the distribution over human parcels that the transport plan connects it to. A human map v over
the 2,094 human parcels becomes a mouse prediction rownorm(pi) v. Reverse-translational
neuroscience asks exactly this question, which mouse circuit corresponds to a human finding, and
until now has had no whole-brain computational tool for it.

We first validated the direction against ground truth. For twelve functional systems with an
established mouse substrate, we routed the human meta-analytic map (Neurosynth association test)
to mouse and ranked mouse structures by the translated value. The ground-truth structure fell in
the top three for nine of twelve systems and cleared a 1,000-rotation spatial spin null for all
twelve (reward to nucleus accumbens and VTA; fear and anxiety to the amygdala; motor to primary
motor cortex and caudoputamen; vision to primary visual cortex; audition to primary auditory
cortex; olfaction to piriform cortex; interoception to the parabrachial nucleus) [ref: Neurosynth;
mouse-substrate reviews already tabulated]. Reverse translation therefore recovers the canonical
mouse substrate of a conserved human function.

We then applied it to a molecular disease substrate. Parkinson's disease is a degeneration of the
dopaminergic nigrostriatal system, and every mouse model of the disease targets that system. We
routed eight human dopamine PET maps to mouse, spanning postsynaptic receptors (D1 SCH23390; D2
and D3 raclopride, fallypride, FLB457) and the presynaptic dopamine transporter (FE-PE2I; FP-CIT,
the clinical DaTscan ligand) [ref: Hansen 2022 neuromaps; tracer primaries]. All eight routed to
the mouse striatum (caudoputamen, nucleus accumbens, olfactory tubercle) and all eight cleared the
spin null (p <= 0.005). The mapping is specific to the dopaminergic system, not a bias toward the
large striatum: cortically organised neurotransmitter maps did not converge there, with the CB1
cannabinoid map routing to somatosensory cortex (p = 0.88), the GABA-A benzodiazepine map to
somatosensory and auditory cortex (p = 0.999), and most serotonergic maps to cortex or colliculus.
Reverse translation thus recovers the striatal dopamine territory that DaTscan images in patients,
identifying the mouse compartment in which the Parkinson's substrate is modelled.

Finally, reverse translation resolves a target at the level of a single symptom dimension. The two
symptom-specific antidepressant TMS circuits, one for dysphoric and one for anxiosomatic symptoms,
are separable in humans but map to overlapping cortical territory [ref: Siddiqi 2020,
10.1176/appi.ajp.2019.19090915]. Routed to mouse, they dissociate. The dysphoric circuit is
medial-prefrontal biased (prelimbic, anterior cingulate, frontal pole), the anxiosomatic circuit
is amygdala and insula biased (basomedial and basolateral amygdala, visceral and agranular
insula), and the between-circuit contrast on a prefrontal-minus-amygdala axis is positive and
significant against a spin null that preserves each circuit's smoothness (C = +0.59, p = 0.0005).
The prescription is concrete: model dysphoric antidepressant targets through mouse medial
prefrontal circuits and anxiosomatic targets through amygdala and insular circuits.

Reverse translation has a boundary, and it is informative. When the human input lacks subcortical
coverage the substrate cannot be recovered: Neurosynth disease maps built from cortically biased
task studies routed Parkinson's to thalamus rather than to the dopaminergic midbrain, and a map
with subcortical signal was required to reach the striatum. The method reads out where in the
mouse brain a human target has a homologue and how confidently, which is the information needed to
choose, or to reject, a mouse model for a given human circuit.

---

## Figure plan (new main figure, e.g. Figure 7, or an Extended Data figure)

- **a. Operator schematic.** Human map -> row-normalised pi -> mouse prediction, one panel, small.
- **b. Functional validation.** Twelve systems x mouse-structure rank; dot or heat strip showing
  ground-truth rank (highlight top-3) and spin-significance. Source: 01_validate.py /
  reverse_translation_validation.json.
- **c. Dopamine / Parkinson.** Left: the eight dopamine maps' top mouse structures on a mouse
  section (striatum lit up). Right: specificity bar, striatal enrichment for dopamine vs
  serotonin vs CB1/GABA-A. Source: 06_neuromaps_substrate.py / reverse_translation_neuromaps.json.
- **d. Symptom dissociation.** Two mouse glass-brains, dysphoric (mPFC) vs anxiosomatic
  (amygdala/insula), with the contrast C and spin p. Source: 07_symptom_dissociation.py /
  reverse_translation_symptom_dissociation.json.

## Citations to add (verify DOIs before submission)
- Neurosynth (Yarkoni 2011) and the twelve mouse-substrate reviews already in
  ground_truth_citations.md.
- neuromaps / Hansen et al. 2022, Nat Neurosci 25:1569-1581 (DOI to verify) + each tracer primary
  (printed by 06_neuromaps_substrate.py).
- Siddiqi et al. 2020, Am J Psychiatry, 10.1176/appi.ajp.2019.19090915 (verified).
- Parkinson mouse-model review for the nigrostriatal substrate (e.g. Blesa & Przedborski 2014) —
  DOI to verify.
- Symptom-substrate anchors: dysphoria/anhedonia to medial prefrontal/subgenual; somatic anxiety
  to amygdala/insula — add 1-2 canonical references, DOIs to verify.
