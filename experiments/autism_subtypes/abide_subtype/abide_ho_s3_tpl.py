"""Step 3. Human templates from the mouse mutation patterns.

Loads the cached human and mouse parcel tables and the canonical coupling, reads the mouse
mutation patterns of TransBrain 2025, and routes them through the coupling. The push-forward
normalises each column by the mass that reaches it over the finite mouse parcels, so every human
parcel takes a weighted mean rather than a weighted sum. Every model pattern is cached over
mouse parcels for the later steps, and the Magel2 template is cached in translated form as the
worked example used by step 4.

Inputs
    outputs/anndata/                                    cached human and mouse AnnData
    outputs/coupling/pi_canonical.npy                   the coupling
    data_external/transbrain_2025/mouse_mutation_pattern.csv
    data_external/transbrain_2025/magel2_mutation_pattern.csv
    experiments/fulcher_2019_multimodal_gradient/01_gradient_validation.py   mouse acronyms

Outputs, in .scratch/abide_ho/
    hx.npy, mx.npy        human and mouse parcel coordinates
    acr.npy               mouse parcel acronyms
    mv_all.npy, mcols.npy every model pattern over mouse parcels, and their names
    tpl_magel2.npy        translated Magel2 template over human parcels
    pi.npy                the coupling used

Run from the repository root.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root():
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / "src" / "otter").is_dir():
            return q
    raise SystemExit("run this from inside the OTTER repository")


R = _repo_root()
WORK = R / ".scratch/abide_ho"
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / 'src'))
t = time.time()
from otter.data import load_cached        # noqa: E402

H, _ = load_cached('human', cache_dir=str(R / 'outputs/anndata'))
M, _ = load_cached('mouse', cache_dir=str(R / 'outputs/anndata'))
print('loaded', time.time() - t, H.shape, M.shape)
sp = importlib.util.spec_from_file_location(
    'fu', R / 'experiments/fulcher_2019_multimodal_gradient/01_gradient_validation.py')
fu = importlib.util.module_from_spec(sp); sp.loader.exec_module(fu)
acr = np.array(fu.load_mouse_parcel_acronyms())
pi = np.load(str(R / 'outputs/coupling/pi_canonical.npy'))
print('acr', acr.shape, 'pi', pi.shape)
hx = H.var[['x', 'y', 'z']].to_numpy(float); mx = M.var[['x', 'y', 'z']].to_numpy(float)
np.save(WORK / 'hx.npy', hx); np.save(WORK / 'mx.npy', mx); np.save(WORK / 'acr.npy', acr)
models = pd.read_csv(R / 'data_external/transbrain_2025/mouse_mutation_pattern.csv', index_col=0)
print('models', list(models.columns))
mag = pd.read_csv(R / 'data_external/transbrain_2025/magel2_mutation_pattern.csv', index_col=0)['Magel2']
mv1 = np.array([models['Magel2'].to_dict().get(a, np.nan) for a in acr], float)
mv2 = np.array([mag.to_dict().get(a, np.nan) for a in acr], float)
print('magel2 from mouse_mutation_pattern finite:', np.isfinite(mv1).sum(),
      ' from magel2 csv:', np.isfinite(mv2).sum(),
      ' identical:', np.allclose(np.nan_to_num(mv1), np.nan_to_num(mv2)))
np.save(WORK / 'mv_all.npy',
        np.array([[models[c].to_dict().get(a, np.nan) for a in acr] for c in models.columns], float))
np.save(WORK / 'mcols.npy', np.array(list(models.columns)))


def route(v):
    ok = np.isfinite(v)
    num = np.nan_to_num(v) @ pi
    den = (pi * ok[:, None]).sum(0)
    o = np.full(pi.shape[1], np.nan)
    g = den > 1e-12
    o[g] = num[g] / den[g]
    return o


tpl = route(mv1)
print('translated Magel2: %d/%d human parcels finite' % (np.isfinite(tpl).sum(), len(tpl)))
np.save(WORK / 'tpl_magel2.npy', tpl)
np.save(WORK / 'pi.npy', pi)
