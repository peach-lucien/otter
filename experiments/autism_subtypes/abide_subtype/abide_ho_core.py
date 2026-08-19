"""Shared setup for the ABIDE case-control steps.

Executed by the scoring steps rather than imported, so that everything it defines is available
to them by name. Loads the ABIDE grand-mean connectivity bundle and its phenotype table, builds
a centroid table column-for-column with the ROI ids of that bundle, and assigns each ABIDE
column to its nearest human parcel. Subcortical ids are matched through their FreeSurfer-style
names, cortical ids through the hemisphere-split label rule of step 1, and the one id that
carries no atlas label is dropped. The assertions state the alignment the table satisfies. Each
column holds the centroid of its own ROI id, and the sign of x agrees with the hemisphere the id
encodes.

A second arm is built alongside it from the bilateral cortical centroids stacked with the
subcortical ones, with the columns taken in order. The two arms are named ``label_matched`` and
``positional``. ``DIAG`` records, for both, the nearest neighbour distances to the human
parcellation and the number of distinct parcels reached.

Defines:
    prep(mask, keepv)     site z-scored connectivity and the phenotype rows for a subject subset
    scorer(Gz, nearv)     a function mapping a human-parcel template to per-subject scores
    run(...)              prep and scorer in one call, kept for step 4
    test(sc, md)          Mann-Whitney U and Cliff's delta between diagnostic groups
    route(v)              push-forward of a mouse-parcel vector through the coupling
    QC                    the motion-passing subject mask
    mvall, mcols, mx      every mouse mutation pattern, their names, and mouse coordinates

Inputs
    $OTTER_ABIDE_BUNDLE (default data_external/abide_ho/)
        abide_G.npz, abide_pheno_small.csv, ho_labels.json                    from step 0
    .scratch/abide_ho/
        cortl_rule.json, cortl_cent.npy, sub_cent.npy, cort_unsplit_cent.npy  from step 1
        hx.npy, mx.npy, pi.npy, mv_all.npy, mcols.npy, tpl_magel2.npy         from step 3

Run from the repository root.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def _repo_root():
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / "src" / "otter").is_dir():
            return q
    raise SystemExit("run this from inside the OTTER repository")


ROOT = _repo_root()
BUNDLE = Path(os.environ.get("OTTER_ABIDE_BUNDLE", ROOT / "data_external/abide_ho"))
WORK = ROOT / ".scratch/abide_ho"
WORK.mkdir(parents=True, exist_ok=True)
if not (BUNDLE / "abide_G.npz").exists():
    raise SystemExit("no abide_G.npz in %s; run abide_ho_s0_bundle.py first" % BUNDLE)

d = np.load(BUNDLE / "abide_G.npz", allow_pickle=True)
ROI = list(map(int, d["roi_ids"])); Graw = d["G"].astype(float); FIDS = list(d["file_ids"])
ph = pd.read_csv(BUNDLE / "abide_pheno_small.csv").set_index("FILE_ID").loc[FIDS].reset_index()
hx = np.load(WORK / "hx.npy"); tpl = np.load(WORK / "tpl_magel2.npy"); pi = np.load(WORK / "pi.npy")
mx = np.load(WORK / "mx.npy")
mvall = np.load(WORK / "mv_all.npy")
mcols = [str(c) for c in np.load(WORK / "mcols.npy")]
QC = ph.qc_ok.to_numpy() == 1

# ---------- label-matched centroid table, column-for-column with ROI ----------
rule = json.load(open(WORK / "cortl_rule.json")); rule = {int(k): v for k, v in rule.items()}
inv = {v: k for k, v in rule.items()}
cl_arr = np.load(WORK / "cortl_cent.npy"); cl = {L: cl_arr[i] for i, L in enumerate(sorted(rule))}
so_arr = np.load(WORK / "sub_cent.npy"); so = {L + 1: so_arr[L] for L in range(len(so_arr))}
labs = json.load(open(BUNDLE / "ho_labels.json"))["sub"]; name2lab = {labs[L]: L for L in range(1, 22)}
fsmap = {10: 'Left Thalamus', 11: 'Left Caudate', 12: 'Left Putamen', 13: 'Left Pallidum',
         17: 'Left Hippocampus', 18: 'Left Amygdala', 26: 'Left Accumbens',
         49: 'Right Thalamus', 50: 'Right Caudate', 51: 'Right Putamen', 52: 'Right Pallidum',
         53: 'Right Hippocampus', 54: 'Right Amygdala', 58: 'Right Accumbens'}
keep = []; cent = []; kind = []; DROPPED = []
for k, i in enumerate(ROI):
    if i in fsmap and fsmap[i] in name2lab:
        keep.append(k); cent.append(so[name2lab[fsmap[i]]]); kind.append('sub')
    elif i in inv:
        keep.append(k); cent.append(cl[inv[i]]); kind.append('cort')
    else:
        DROPPED.append(i)
keep = np.array(keep); cent = np.array(cent); kind = np.array(kind)
ROIk = [ROI[k] for k in keep]
# ---- alignment: every column holds the centroid of its own ROI id ----
assert len(ROIk) == len(cent) == len(keep), 'centroid table not column-aligned to ROI ids'
for j, k in enumerate(keep):
    i = ROIk[j]
    if i in fsmap:
        exp = so[name2lab[fsmap[i]]]; assert np.allclose(cent[j], exp), (i, 'sub misaligned')
        assert (cent[j][0] < 0) == (fsmap[i].startswith('Left')), (i, 'sub hemisphere mismatch')
    else:
        assert rule[inv[i]] == i, (i, 'cortl label->id rule violated')
        assert (cent[j][0] < 0) == (i % 100 == 1), (i, 'cortical hemisphere code mismatch')
    assert ROI[k] == i
print('ALIGNMENT ASSERTION PASSED: %d columns, ids[0:3]=%s, dropped=%s' % (len(ROIk), ROIk[:3], DROPPED))

D = np.sqrt(((cent[:, None, :] - hx[None, :, :]) ** 2).sum(-1))
near = D.argmin(1); nd = D[np.arange(len(near)), near]
CM = kind == 'cort'
DIAG = {'label_matched': {'n_cent': len(near), 'median_nn_mm': float(np.median(nd)),
        'cort_median_nn_mm': float(np.median(nd[CM])),
        'cort_distinct_parcels': int(len(set(near[CM].tolist()))), 'cort_n': int(CM.sum()),
        'sub_distinct_parcels': int(len(set(near[~CM].tolist()))), 'sub_n': int((~CM).sum())}}

# ---------- positional arm: bilateral cortical centroids, columns taken in order ----------
oc = np.load(WORK / "cort_unsplit_cent.npy"); ocent = np.vstack([oc, so_arr])
Do = np.sqrt(((ocent[:, None, :] - hx[None, :, :]) ** 2).sum(-1))
nearo = Do.argmin(1); ndo = Do[np.arange(len(nearo)), nearo]
DIAG['positional'] = {'n_cent': len(nearo), 'median_nn_mm': float(np.median(ndo)),
                      'cort_median_nn_mm': float(np.median(ndo[:48])),
                      'cort_distinct_parcels': int(len(set(nearo[:48].tolist()))), 'cort_n': 48,
                      'sub_distinct_parcels': int(len(set(nearo[48:].tolist()))), 'sub_n': len(so_arr)}
# agreement between the 48 bilateral assignments and the label-matched ones
corr_by_cort = {}
for j, i in enumerate(ROIk):
    if i >= 100:
        corr_by_cort.setdefault(i // 100, []).append(near[j])
agree = sum(1 for c in range(1, 49) if nearo[c - 1] in corr_by_cort.get(c, []))
DIAG['positional']['agree_with_label_matched_of_48'] = int(agree)
POSITIONAL_COLS = np.arange(len(ocent))   # the positional arm takes its columns in order


def route(v, coupling=None):
    """Push a mouse-parcel vector through the coupling, normalised by the mass each column receives."""
    P = pi if coupling is None else coupling
    ok = np.isfinite(v)
    num = np.nan_to_num(v) @ P
    den = (P * ok[:, None]).sum(0)
    o = np.full(P.shape[1], np.nan)
    g = den > 1e-12
    o[g] = num[g] / den[g]
    return o


def prep(mask, keepv):
    """Site-wise z-scored connectivity for a subject subset, against that site's controls."""
    G = Graw[np.ix_(mask, keepv)]
    md = ph[mask].reset_index(drop=True)
    Gz = np.full_like(G, np.nan)
    for s in md.SITE_ID.unique():
        m = (md.SITE_ID == s).to_numpy()
        c = m & (md.DX_GROUP.to_numpy() == 2)
        if c.sum() < 3:
            continue
        Gz[m] = (G[m] - np.nanmean(G[c], 0)) / (np.nanstd(G[c], 0) + 1e-9)
    return Gz, md


def scorer(Gz, nearv):
    """Return f(template over human parcels) -> (per-subject score, number of ROIs used)."""
    def f(tplfull):
        T = tplfull[nearv]
        v = np.isfinite(T) & np.isfinite(Gz).all(0)
        Tz = (T[v] - np.nanmean(T[v])) / np.nanstd(T[v])
        A = Gz[:, v]
        sd = A.std(1)
        Az = (A - A.mean(1, keepdims=True)) / np.where(sd > 1e-9, sd, np.nan)[:, None]
        return Az @ ((Tz - Tz.mean()) / Tz.std()) / len(Tz), int(v.sum())
    return f


def run(mask, tplfull, nearv, keepv, label=None):
    """prep and scorer in one call."""
    Gz, md = prep(mask, keepv)
    sc, nr = scorer(Gz, nearv)(tplfull)
    return sc, nr, md


def test(sc, md):
    a = (md.DX_GROUP.to_numpy() == 1) & np.isfinite(sc)
    c = (md.DX_GROUP.to_numpy() == 2) & np.isfinite(sc)
    u, p = mannwhitneyu(sc[a], sc[c])
    return dict(n_asd=int(a.sum()), n_ctrl=int(c.sum()), u=float(u), p=float(p),
                cliffs=float(2 * u / (a.sum() * c.sum()) - 1),
                asd_mean=float(sc[a].mean()), asd_sd=float(sc[a].std()),
                ctrl_mean=float(sc[c].mean()), ctrl_sd=float(sc[c].std()),
                delta_mean=float(sc[a].mean() - sc[c].mean()))
