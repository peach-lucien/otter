# TransBrain 2025, sibling-method benchmark

A comparison of OTTER against TransBrain on a shared homology benchmark and on
shared phenotypes.

## Why this experiment

[Huang et al. 2025, Nature Methods](https://doi.org/10.1038/s41592-025-02961-3),
[TransBrain](https://github.com/ibpshangzheng/transbrain), is a published
mouse↔human phenotype-translation framework. It works at region level (68 mouse
regions; Brainnetome, DK and AAL human atlases) via graph embeddings and dual
regression. OTTER produces a soft parcel-level coupling π by Fused
Gromov-Wasserstein optimal transport. The two rest on different principles, and
this experiment measures how they compare.

Unless stated otherwise, every number below is read from
`outputs/logs/transbrain_2025_benchmark.json` or
`outputs/logs/transbrain_roundtrip_maps.json`. Both record the canonical
coupling `outputs/coupling/pi_canonical.npy`, sha256
`bb4cae00cbca9f16c6f9cfca3b0124292b41d81643e2ef5d5511686b20f9df77`.

## Result

`outputs/figures/transbrain_2025_benchmark.png`, 3 panels.

**Part A. Homology benchmark.** TransBrain ships a literature-curated set of
mouse↔human homologous region pairs (`data_external/transbrain_2025/homo_cortex.csv`,
`data_external/transbrain_2025/homo_subcortex.csv`), a benchmark OTTER has not
been fitted to, and independent of the Garin anchors and the Beauchamp set.
Routing OTTER's π for the 17 scorable cortical mouse regions, the
literature-homolog Brainnetome region lands in OTTER's top-1 35.3 % of the time,
top-3 64.7 % and top-5 76.5 %, on an atlas of 127 Brainnetome regions where
top-1 chance is 1.9 %. The permuted-π null reaches top-3 2.5 %, and no null draw
out of 200 matched the observed top-3 (empirical p = 0.0). On the
resolution-fair metric, OTTER's predicted human centroid sits 16.21 mm from the
literature homolog against 39.53 mm for the permuted-π null (empirical p = 0.0).

The 7-region subcortical benchmark separates into two results. On rank it is at
chance: top-1 and top-3 are both 0.0, and the top-3 empirical p against the
permuted-π null is 1.0. Top-5 is 42.9 %, for which the log records no null. On
distance it is not at chance: the predicted centroid sits 10.12 mm from the
literature homolog against 24.42 mm for the null (empirical p = 0.0).

**Part B. Head-to-head.** The same mouse phenotype translated by both methods
and compared at Brainnetome-region level.

| Phenotype | OTTER vs human | TransBrain vs human | OTTER vs TransBrain |
|---|---:|---:|---:|
| resting-fMRI gradient, 101 BN cortical regions | \|r\| = 0.562 | \|r\| = 0.517 | \|r\| = 0.826 |
| Magel2 mutation pattern, 122 BN regions, 233 individuals | not scored | not scored | r = 0.584 (maps), r = 0.606 (risk scores) |

Both methods recover the human gradient. OTTER scores 0.562 and TransBrain
0.517. The log records no test of that difference, so the two are reported as
point estimates. The two translated gradients agree with each other at 0.826.

The Magel2 run scores no human reference map, so the only comparison available
is between the methods. Their translated maps correlate at r = 0.584. The
per-individual ASD risk-score workflow (TransBrain's own case 3, reproduced)
gives r = 0.606 across 233 individuals.

## Round-trip consistency

Round-tripping a phenotype mouse→human→mouse, a metric that needs no ground
truth. From `outputs/logs/transbrain_roundtrip_maps.json`, which scores both
methods over the identical 52 mouse regions in which each phenotype is measured.

| Phenotype | OTTER | TransBrain |
|---|---:|---:|
| resting-fMRI gradient | r = 0.968 | r = 0.891 |
| anterior-insula optogenetic circuit | r = 0.863 | r = 0.821 |
| Magel2 mutation pattern | r = 0.910 | r = 0.834 |

OTTER's value is higher on all three. The log records no test of these
differences.

## Anterior-insula optogenetic circuit

`05_aiopto_headtohead.py` translates TransBrain's own anterior-insula
optogenetic map with both methods and scores each translated human map with the
same salience-versus-rest enrichment metric, on the shared support of 1,635
human parcels. From `outputs/logs/section6_transbrain_aiopto.json`, which
records the canonical π sha256. OTTER's enrichment is z = 0.867, above its
permutation null (1,000 permutations of the mouse region-to-value assignment,
p = 0.016). TransBrain's is z = 0.277, which its own null does not separate
(p = 0.228). The log records no test of the difference between the two methods.

## Advanced comparison

`03_transbrain_advanced.py` writes `outputs/logs/transbrain_2025_advanced.json`.
That log records no π sha256, so the coupling that produced it cannot be
confirmed from the log alone. Agreement here is measured as the distance between
OTTER's and TransBrain's top human Brainnetome region for a given mouse region.

**Trust-stratified agreement.** Over 52 mouse regions, top-region distance and
OTTER's per-parcel trust score correlate at r = -0.034 (Spearman -0.057). Tier
means run from 19.2 mm (anchored_only) to 30.9 mm (anchored_and_validated), with
validated_only 24.6 mm, structural 28.6 mm and low_evidence 29.8 mm, and do not
order by tier. OTTER's trust map does not predict where the two methods agree.

**Consensus and disagreement map.** Mouse regions ranked by OTTER↔TransBrain
top-region distance. Four regions share the same top Brainnetome region (ACAd,
VISpm, PERI, RT). The largest distances are PL (82.0 mm), VISam (77.1 mm),
SSp-n (70.8 mm) and CA2 (65.3 mm).

## Method

1. Build an OTTER-parcel → Brainnetome-region map by sampling TransBrain's BN
   atlas at each OTTER human parcel's MNI centroid (3×3×3 fallback).
2. Part A, for each benchmarked mouse region, route π and rank BN regions by
   received mass; also measure the predicted-centroid distance to the
   literature homolog. Permuted-π null, 200 trials.
3. Part B, translate the mouse principal gradient and the Magel2 mutation
   pattern with both methods; compare at BN-region level.

## Files

| File | What |
|---|---|
| `01_transbrain_benchmark.py` | Homology benchmark + both head-to-heads |
| `02_plot.py` | 3-panel figure (benchmark + head-to-heads) |
| `03_transbrain_advanced.py` | Trust-stratified agreement, consensus map |
| `05_aiopto_headtohead.py` | Anterior-insula optogenetic head-to-head |
| `06_bn_distributions.py` | Per-region Brainnetome distributions for both methods; writes `outputs/logs/transbrain_bn_distributions.json` |
| `07_benchmark_summary.py` | AUROC, top-k, mass-in-region, sharpness and win counts; writes `outputs/logs/transbrain_benchmark_summary.json` |
| `08_roundtrip_maps.py` | Round-trip maps and correlations for both methods; writes `outputs/logs/transbrain_roundtrip_maps.json` |
| `09_localization_distributions.py` | Predicted distributions for five example regions; writes `outputs/logs/localization_distributions.json` |
| `README.md` | This file |

`notebooks/06_vs_transbrain.ipynb` and the repository's number-checking tools
read the four logs written by `06` to `09`.

## Reproduce

```bash
pip install transbrain          # Apache-2.0; supplies the human BN atlas
PYTHONPATH=src python experiments/transbrain_2025_benchmark/01_transbrain_benchmark.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/02_plot.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/03_transbrain_advanced.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/05_aiopto_headtohead.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/06_bn_distributions.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/07_benchmark_summary.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/08_roundtrip_maps.py
PYTHONPATH=src python experiments/transbrain_2025_benchmark/09_localization_distributions.py
```

The gradient head-to-head depends on the Margulies experiment's output
(`outputs/logs/margulies_2016_gradient.json`). TransBrain benchmark and case
data are staged in `data_external/transbrain_2025/`.

`07_benchmark_summary.py` reads `outputs/logs/transbrain_bn_distributions.json`,
which is committed, and runs on a fresh clone. The other scripts read
`data_external/`, `outputs/coupling/` and `outputs/anndata/`, which are gitignored
and come from the Zenodo reproduce bundle. Run `06_bn_distributions.py` before
`07_benchmark_summary.py`.
