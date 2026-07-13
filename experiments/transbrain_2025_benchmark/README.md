# TransBrain 2025, sibling-method benchmark

A **methods-landscape** comparison, not a validation "pass". It
positions HOMER against the current state-of-the-art mouse↔human translator.

## Why this experiment

[Huang et al. 2025, Nature Methods](https://doi.org/10.1038/s41592-025-02961-3),
[TransBrain](https://github.com/ibpshangzheng/transbrain), is a published
mouse↔human phenotype-translation framework, a direct sibling of HOMER. It
works at **region level** (68-region mouse atlas; Brainnetome / DK / AAL human
atlases) via graph embeddings + dual regression, where HOMER produces a soft
parcel-level coupling π via Fused Gromov-Wasserstein optimal transport. Two
methods, different principles, this experiment asks how they compare.

## Result

`outputs/figures/transbrain_2025_benchmark.png`, 3 panels.

**Part A. Homology benchmark.** TransBrain ships a literature-curated set of
classic mouse↔human homologous region pairs (`homo_cortex.csv`,
`homo_subcortex.csv`), a benchmark HOMER has never seen, independent of the
Garin anchors and the Beauchamp set. Routing HOMER's π for the 17 scorable
cortical mouse regions: the literature-homolog Brainnetome region lands in
HOMER's **top-3 41 %** of the time (permuted-π null 4 %, p < 0.001) and top-5
47 %, modest on a fine 127-region atlas. The resolution-fair metric is
clearer: HOMER's predicted human centroid sits **25.3 mm** from the literature
homolog vs **39.8 mm** for the null (p < 0.001). HOMER places mouse regions in
the right neighbourhood, and 25.3 mm is squarely within its own stated
~25–45 mm resolution, but does not pinpoint the exact Brainnetome parcel. The
7-region subcortical benchmark comes out at chance.

**Part B. Head-to-head.** The same mouse phenotype translated by both methods,
compared at Brainnetome-region level:

| Phenotype | HOMER vs human | TransBrain vs human | HOMER ↔ TransBrain |
|---|---:|---:|---:|
| resting-fMRI gradient | \|r\| = 0.393 | \|r\| = 0.463 | \|r\| = 0.23 |
| Magel2 autism pattern | | | r = 0.10 (maps); 0.05 (risk scores) |

On the smooth gradient both methods recover the human reference, with
TransBrain, a tool purpose-built for region-level phenotype translation
scoring higher. On the noisy Magel2 autism mutation pattern the two methods
diverge. The per-individual ASD risk-score workflow (TransBrain's own case 3,
reproduced) gives near-zero concordance: the autism phenotype is noisy for
both methods.

**Assessment.** HOMER and TransBrain are different tools that agree
only moderately (\|r\| ≈ 0.2–0.3). TransBrain is stronger for region-level
phenotype translation, its home turf. HOMER's complementary contribution is a
*soft, parcel-level* coupling with per-parcel trust tiers and explicit anchor
supervision. This experiment is best read as positioning, not a contest.

## Advanced comparison

Four follow-ups (`03_transbrain_advanced.py`, figure `transbrain_2025_advanced.png`)
dig past the average.

**Bidirectional cycle-consistency.** Round-tripping a phenotype
mouse→human→mouse, an even-handed, ground-truth-free metric with no home-turf
advantage. HOMER recovers the original at **0.98 / 0.95 / 0.97** (gradient, optogenetic
circuit, Magel2 autism pattern) versus **0.89 / 0.82 / 0.83** for TransBrain, both
scored over the same 52 mouse regions in which the phenotype is measured.

> ⚠️ **Corrected July 2026.** This previously read "0.81–0.91 for TransBrain". Those
> numbers scored HOMER on the 52 regions its parcellation covers but TransBrain on all
> 68 of `Config.MOUSE_REGIONS` — 16 of them **mean-filled** for the gradient. The two
> were not comparable, and the error did not even bias consistently (it inflated
> TransBrain on the optogenetic map, 0.82 → 0.91, and deflated it on the gradient,
> 0.89 → 0.87). Both are now scored on the identical region set. HOMER still leads all
> three; the margin is narrowest on the smooth gradient (+0.09) and widest on the two
> focal maps (+0.13 each). HOMER's soft optimal-transport coupling is more internally
coherent in both directions, a genuine HOMER strength.

**Optogenetic circuit → human cognition.** Reproducing TransBrain's Case 2: a
mouse anterior-insula optogenetic circuit routed through π and decoded against
114 Neurosynth cognitive-term maps. HOMER's top terms emphasise language /
cognitive-control; TransBrain's emphasise interoception / reward, both genuine
insula functions, overlapping 2/10.

**Trust-stratified agreement, a negative result.** HOMER↔TransBrain agreement
does *not* track HOMER's trust tiers (r ≈ 0, flat across all five). HOMER's
trust map reflects its own anchor/validation evidence, not inter-method
consensus, so the methods' disagreement is not explained by HOMER's confidence.

**Consensus / disagreement map.** Mouse regions ranked by HOMER↔TransBrain
top-region distance flag which homologies the two methods concur on versus
contest.

## TransBrain's own example notebooks

We ran all five TransBrain tutorial notebooks. `gene_mutations.ipynb` (autism
case) executes fully end-to-end. The core `SpeciesTrans` translation API works
and is used directly in this experiment. The remaining notebooks' translation
cells execute but their visualisation cells need surface-plotting dependencies
not present in our environment, and `fMRI_gradients.ipynb` needs gradient
NIfTI files not bundled in the TransBrain repo.

## Method

1. Build a HOMER-parcel → Brainnetome-region map by sampling TransBrain's BN
   atlas at each HOMER human parcel's MNI centroid (3×3×3 fallback).
2. Part A, for each benchmarked mouse region, route π and rank BN regions by
   received mass; also measure the predicted-centroid distance to the
   literature homolog. Permuted-π null (200 trials).
3. Part B, translate the mouse principal gradient and the Magel2 mutation
   pattern with both methods; compare at BN-region level.

## Files

| File | What |
|---|---|
| `01_transbrain_benchmark.py` | Homology benchmark + both head-to-heads |
| `02_plot.py` | 3-panel figure (benchmark + head-to-heads) |
| `03_transbrain_advanced.py` | Cycle-consistency, optogenetic decode, trust-stratified, consensus map |
| `04_advanced_plot.py` | 3-panel advanced figure |
| `README.md` | This file |

## Reproduce

```bash
pip install transbrain          # Apache-2.0; supplies the human BN atlas
PYTHONPATH=src python experiments/transbrain_2025_benchmark/01_transbrain_benchmark.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/02_plot.py
# advanced (the optogenetic decode needs the TransBrain repo's neurosynth_data;
# point NEUROSYNTH_DIR at it)
PYTHONPATH=src python experiments/transbrain_2025_benchmark/03_transbrain_advanced.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/04_advanced_plot.py
```

Depends on the Margulies experiment's output (`outputs/logs/margulies_2016_gradient.json`)
for the gradient head-to-head. TransBrain benchmark + case data are staged in
`data_external/transbrain_2025/`.

