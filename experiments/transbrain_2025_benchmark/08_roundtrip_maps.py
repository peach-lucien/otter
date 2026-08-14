#!/usr/bin/env python3
"""Cache the mouse phenotype maps through the mouse->human->mouse round trip, for
both OTTER and TransBrain, so we can paint 'original vs recovered' on the mouse
brain (panel B). Reproduces the cycle-consistency of 03_transbrain_advanced.py.

Requires the third-party `transbrain` package, plus the gitignored `data_external/`
and `outputs/coupling/` inputs from the Zenodo reproduce bundle.

Run: cd otter && PYTHONPATH=src python experiments/transbrain_2025_benchmark/08_roundtrip_maps.py
Writes outputs/logs/transbrain_roundtrip_maps.json (per-parcel arrays, 1864 long).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr
warnings.filterwarnings('ignore'); sys.path.insert(0, 'src')
from transbrain.config import Config
from transbrain.trans import SpeciesTrans
ROOT = Path('.').resolve()
DATA = ROOT / 'data_external' / 'transbrain_2025'
from otter.data import load_pi, pi_provenance   # never hardcode which pi is canonical
pi = load_pi()
mm = json.loads(Path('data_external/mouse_sc_meta.json').read_text())
parcel_acr = np.array([mm['structure_acronyms'][i] for i in mm['node_struct_idx']])
MOUSE_REGIONS = list(Config.MOUSE_CORTICAL) + list(Config.MOUSE_SUBCORTICAL)
st = SpeciesTrans(atlas_type='bn')

def route_fwd(m, pi, mask):
    num = m[mask] @ pi[mask, :]; den = pi[mask, :].sum(0)
    out = np.full(pi.shape[1], np.nan); ok = den > 1e-12; out[ok] = num[ok]/den[ok]; return out
def route_rev(h, pi, mask):
    num = pi[:, mask] @ h[mask]; den = pi[:, mask].sum(1)
    out = np.full(pi.shape[0], np.nan); ok = den > 1e-12; out[ok] = num[ok]/den[ok]; return out
def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(pearsonr(a[m], b[m])[0]) if m.sum() > 3 else float('nan')

mg = json.loads(Path('outputs/logs/margulies_2016_gradient.json').read_text())
grad = {}
for a, v in zip(parcel_acr, np.array(mg['mouse_gradient'])):
    if a in MOUSE_REGIONS: grad.setdefault(a, []).append(v)
phenos = {'gradient': {a: float(np.mean(v)) for a, v in grad.items()},
          'AI_opto': pd.read_csv(DATA/'ai_opto.csv', index_col=0).iloc[:, 0].to_dict(),
          'Magel2': pd.read_csv(DATA/'magel2_mutation_pattern.csv', index_col=0)['Magel2'].to_dict()}

out = {}
for name, ph in phenos.items():
    mvec = np.array([ph.get(a, np.nan) for a in parcel_acr])          # original, per parcel
    fwd = route_fwd(mvec, pi, np.isfinite(mvec))
    oback = route_rev(fwd, pi, np.isfinite(fwd))                       # OTTER recovered, per parcel
    full = {a: ph.get(a, np.nan) for a in MOUSE_REGIONS}
    fv = np.nanmean(list(full.values())); full = {a: (v if np.isfinite(v) else fv) for a, v in full.items()}
    tb_back = st.human_to_mouse(st.mouse_to_human(pd.DataFrame({name: full}), region_type='all'), region_type='all')
    tbmap = tb_back.iloc[:, 0].to_dict()
    tbvec = np.array([tbmap.get(a, np.nan) for a in parcel_acr])       # TransBrain recovered, per parcel

    # ---- MATCHED region-level scoring -------------------------------------------------
    # Both methods are scored on the same regions, or the comparison is not a comparison.
    #
    # Two region sets are available and they are not interchangeable. OTTER's parcellation
    # covers 52 of Config.MOUSE_REGIONS; the other 16 come back NaN. TransBrain returns all
    # 68. Scoring each method on its own set compares different quantities, and for the
    # gradient phenotype 16 of the 68 have no measured value at all: they are mean-filled
    # above so TransBrain can be given a complete vector, and a mean-filled entry is not
    # data to score against.
    #
    # The scored set is therefore the regions where the phenotype is measured and that
    # OTTER's parcellation covers, identical for both methods.
    scored = [a for a in MOUSE_REGIONS
              if a in ph and np.isfinite(ph[a]) and (parcel_acr == a).any()]
    orig_r = np.array([ph[a] for a in scored])
    otter_r = np.array([np.nanmean(oback[parcel_acr == a]) for a in scored])
    tb_r = np.array([tbmap.get(a, np.nan) for a in scored])
    r_o, r_t = corr(orig_r, otter_r), corr(orig_r, tb_r)

    out[name] = dict(original=list(map(float, np.nan_to_num(mvec, nan=np.nan))),
                     otter=list(map(float, oback)), transbrain=list(map(float, tbvec)),
                     r_otter=r_o, r_transbrain=r_t,
                     n_regions_scored=len(scored), regions_scored=scored,
                     _scoring=('region-level Pearson r over the regions where the phenotype is '
                               'measured AND OTTER has parcels; identical region set for both '
                               'methods. See the scoring note in 08_roundtrip_maps.py.'))
    print(f'{name:9s} round-trip r ({len(scored)} regions):  '
          f'OTTER {r_o:+.3f}   TransBrain {r_t:+.3f}   margin {r_o - r_t:+.3f}')
# Figure 4b's round-trip correlations come from this file, so the coupling that
# produced them is recorded alongside them.
out.update(pi_provenance())
json.dump(out, open('outputs/logs/transbrain_roundtrip_maps.json', 'w'))
print('saved -> outputs/logs/transbrain_roundtrip_maps.json')
