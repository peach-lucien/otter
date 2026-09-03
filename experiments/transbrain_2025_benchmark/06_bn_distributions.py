#!/usr/bin/env python3
"""Localisation on a shared atlas: put OTTER and TransBrain on the same BN atlas and
score how well each ranks the true human homologue region (reduction-free, resolution-
matched, hemisphere-agnostic). Caches per-region distributions for scoring in-sandbox.

For each benchmark mouse region it stores, over a common list of BN regions:
  - otter_w : OTTER parcel coupling aggregated (summed) into BN regions
  - tb_w    : TransBrain's BN-region weights (clipped >=0, normalised)
  - true    : indices (into the common BN list) of the literature homologue region(s)

Requires the third-party `transbrain` package, plus the gitignored `data_external/`,
`outputs/coupling/` and `outputs/anndata/` inputs from the Zenodo reproduce bundle.

Run: cd otter && PYTHONPATH=src python experiments/transbrain_2025_benchmark/06_bn_distributions.py
Writes outputs/logs/transbrain_bn_distributions.json
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

from otter.data import load_pi, pi_provenance   # never hardcode which pi is canonical
pi = load_pi()
mm = json.loads(Path('data_external/mouse_sc_meta.json').read_text())
parcel_acr = np.array([mm['structure_acronyms'][i] for i in mm['node_struct_idx']])
H, _ = load_cached('human', cache_dir='outputs/anndata')
bn_id, id2name, centroid = tb01.load_bn_atlas(H.var)         # bn_id: BN region id per human parcel
bn_id = np.asarray(bn_id)
mr_all = list(Config.MOUSE_CORTICAL) + list(Config.MOUSE_SUBCORTICAL)
tb_mat = SpeciesTrans(atlas_type='bn').mouse_to_human(pd.DataFrame(np.eye(len(mr_all)), index=mr_all, columns=mr_all), region_type='all')

# common BN region list = TransBrain's rows that we also have an OTTER aggregation for
bn_names = [n for n in tb_mat.index if n in centroid]
name2col = {n: k for k, n in enumerate(bn_names)}
# map each human parcel to a column in bn_names (via id2name), -1 if not in the common list
parcel_to_col = np.array([name2col.get(id2name.get(int(b), None), -1) for b in bn_id])

out = {}
for csv in ['homo_cortex.csv', 'homo_subcortex.csv']:
    b = pd.read_csv(f'data_external/transbrain_2025/{csv}', index_col=0)
    hom = {}
    for _, r in b.iterrows(): hom.setdefault(r['mouse_region'], set()).add(r['human_region'])
    for mr, hs in hom.items():
        idx = np.where(parcel_acr == mr)[0]
        hs = [h for h in hs if h in name2col]
        if len(idx) == 0 or not hs or mr not in tb_mat.columns: continue
        # OTTER: aggregate parcel coupling into BN regions
        pw = pi[idx].sum(0)
        otter_w = np.zeros(len(bn_names))
        m = parcel_to_col >= 0
        np.add.at(otter_w, parcel_to_col[m], pw[m])
        if otter_w.sum() > 0: otter_w = otter_w / otter_w.sum()
        # TransBrain
        tv = tb_mat[mr]; tw = np.array([max(float(tv.get(n, 0.0)), 0.0) for n in bn_names])
        if tw.sum() > 0: tw = tw / tw.sum()
        out[mr] = dict(otter_w=list(map(float, otter_w)), tb_w=list(map(float, tw)),
                       true=[name2col[h] for h in hs])
# Record the provenance of the coupling used to construct these distributions.
json.dump(dict(bn_names=bn_names, regions=out, **pi_provenance()),
          open('outputs/logs/transbrain_bn_distributions.json', 'w'))
print(f"cached BN distributions for {len(out)} regions over {len(bn_names)} BN regions "
      f"-> outputs/logs/transbrain_bn_distributions.json")
