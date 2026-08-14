#!/usr/bin/env python3
"""Cache the full predicted DISTRIBUTIONS for a few example homologues, for both
OTTER and TransBrain, so both can be drawn on the same human parcel cloud.

Requires the third-party `transbrain` package, plus the gitignored `data_external/`,
`outputs/coupling/` and `outputs/anndata/` inputs from the Zenodo reproduce bundle.

WHY TransBrain IS STORED AS tb_parcel AND NOT AS REGION CENTROIDS. TransBrain's
bundled Brainnetome is hemisphere-merged (127 labels; verified 127 labels / 127
label-table rows / 127 unique anatomical names), so every region spans both sides
and its voxel centroid falls at x = 0 by symmetry. Storing the top-K region
centroids and plotting them as markers is arithmetically right and visually
indefensible: all 127 regions land on the midline, which reads as TransBrain
predicting midline structures rather than bilateral regions. Coordinates are also
invisible to every statistic here, because AUROC, the gradient r and the
round-trip r all run through aggregate_bn on region WEIGHTS.

tb_parcel is TransBrain's region weight painted onto the same human parcels OTTER
is scored on, via the bn_id lookup. Constant within a region, which is what a
region-level method actually asserts. The weight is NOT divided by parcel count,
or large regions would render dim and the panel would confuse weight with region
size.

Run: cd otter && PYTHONPATH=src python experiments/transbrain_2025_benchmark/09_localization_distributions.py
Writes outputs/logs/localization_distributions.json
"""
import sys, json, importlib.util, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, 'src')
spec = importlib.util.spec_from_file_location('tb01', 'experiments/transbrain_2025_benchmark/01_transbrain_benchmark.py')
tb01 = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb01)
from transbrain.config import Config
from transbrain.trans import SpeciesTrans
from otter.data import load_cached

EXAMPLES = ['VISp', 'ORBl', 'MOp', 'CA2', 'SSp-m']   # spatially spread; edit if desired
TOPK = 40

from otter.data import load_pi, pi_provenance   # never hardcode which pi is canonical
pi = load_pi()
mm = json.loads(Path('data_external/mouse_sc_meta.json').read_text())
parcel_acr = np.array([mm['structure_acronyms'][i] for i in mm['node_struct_idx']])
H, _ = load_cached('human', cache_dir='outputs/anndata'); hxyz = H.var[['x', 'y', 'z']].to_numpy(float)
bn_id, id2name, centroid = tb01.load_bn_atlas(H.var)
mr_all = list(Config.MOUSE_CORTICAL) + list(Config.MOUSE_SUBCORTICAL)
tb_mat = SpeciesTrans(atlas_type='bn').mouse_to_human(pd.DataFrame(np.eye(len(mr_all)), index=mr_all, columns=mr_all), region_type='all')

out = {}
for csv in ['homo_cortex.csv', 'homo_subcortex.csv']:
    b = pd.read_csv(f'data_external/transbrain_2025/{csv}', index_col=0)
    hom = {}
    for _, r in b.iterrows(): hom.setdefault(r['mouse_region'], set()).add(r['human_region'])
    for mr, hs in hom.items():
        if mr not in EXAMPLES: continue
        idx = np.where(parcel_acr == mr)[0]; hs = [h for h in hs if h in centroid]
        if len(idx) == 0 or not hs or mr not in tb_mat.columns: continue
        truth = np.mean([centroid[h] for h in hs], 0)
        # OTTER distribution over human parcels
        ow = pi[idx].sum(0); ow = ow / ow.sum(); oo = np.argsort(-ow)[:TOPK]
        otter_top = [[float(hxyz[i, 0]), float(hxyz[i, 1]), float(hxyz[i, 2]), float(ow[i])] for i in oo]
        # TransBrain distribution over BN regions
        tv = tb_mat[mr]; names = [r for r in tv.index if r in centroid]
        tw = np.clip(tv[names].values.astype(float), 0, None); tw = tw / tw.sum() if tw.sum() > 0 else tw
        # TransBrain as a piecewise-constant map over the SAME parcels OTTER uses.
        # Every parcel inside a BN region carries that region's whole weight, so the
        # panel shows what a region-level method claims: one value for the region.
        name_w = dict(zip(names, tw.tolist()))
        tb_parcel = np.zeros(len(hxyz))
        for rid in np.unique(bn_id):
            if not rid:
                continue
            w = name_w.get(id2name.get(int(rid)))
            if w is not None:
                tb_parcel[bn_id == rid] = w
        out[mr] = dict(truth=list(map(float, truth)),
                       otter_top=otter_top,
                       tb_parcel=[float(v) for v in tb_parcel])
out['_meta'] = dict(
    **pi_provenance(),          # returns {"pi_file", "pi_sha256"}
    n_bn_labels=int(len({int(r) for r in np.unique(bn_id) if r})),
    n_human_parcels=int(len(hxyz)),
    note=("tb_parcel is TransBrain's BN-region weight painted onto OTTER's human "
          "parcels, constant within a region. Region centroids are not stored: the "
          "bundled BN atlas is hemisphere-merged, so every centroid sits at x=0."),
)
json.dump(out, open('outputs/logs/localization_distributions.json', 'w'), indent=1)
print('cached distributions for:', [k for k in out if not k.startswith('_')],
      '-> outputs/logs/localization_distributions.json')
print('pi:', out['_meta']['pi_file'], out['_meta']['pi_sha256'][:12])
