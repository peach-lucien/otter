"""Reachability under a spin null, with a specificity check against a hierarchy map.

A plain permutation of burden across regions destroys spatial autocorrelation and inflates
significance, since both coverage and disease maps are smooth. The spin null used here rotates the
coverage map on the sphere and preserves autocorrelation. This also tests whether the disease burden
aligns with reconstruction-coverage MORE than with a generic myelin/hierarchy map -
if myelin does just as well, the signal is 'disorders hit association cortex', not OTTER-specific.

reachability(disorder) = burden-weighted mean of a reference-map PERCENTILE across 52 hemi-regions,
burden = max(0,-d). <50 = burden on low-reference cortex. Spin p (two-sided on |obs-50|).
References compared: reconstruction-coverage (FC+SC, pi_canonical) vs T1w/T2w myelin (hierarchy).

Writes outputs/logs/section6_reachability_spin.json
"""
from __future__ import annotations
import csv, glob, json, sys
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import abagen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance  # noqa: E402
from otter.eval.nulls import _haar_rotation             # noqa: E402


def recon(pi, Mc, Hc):
    ph = pi.sum(0); pit = pi / np.maximum(ph, 1e-300); pred = pit.T @ Mc @ pit
    n = pred.shape[0]; out = np.full(n, np.nan)
    for j in range(n):
        a = pred[j].copy(); b = Hc[j].copy(); a[j] = np.nan; b[j] = np.nan
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10 and a[ok].std() > 1e-9: out[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


def enig(p):
    o = {}
    for row in csv.DictReader(open(p)):
        s = row["Structure"]
        if "_" not in s: continue
        h, r = s.split("_", 1)
        try: v = float(row["d_icv"])
        except: continue
        if h in ("L", "R"): o[(h, r.strip().lower())] = v
    return o


def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    Mfc = np.asarray(M.uns["fc_mean"], float); Hfc = np.asarray(H.uns["fc_mean"], float)
    Msc = np.log1p(np.maximum(np.load(ROOT / "data_external/mouse_sc.npy").astype(float), 0))
    Hsc = np.log1p(np.maximum(np.load(ROOT / "data_external/human_sc.npy").astype(float), 0))
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    pi = load_pi()
    rc = recon(pi, 0.7 * Mfc + 0.3 * Msc, 0.7 * Hfc + 0.3 * Hsc)
    mye = np.asarray(json.loads((ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())["myelin_per_parcel"], float)

    at = abagen.fetch_desikan_killiany(); img = nib.load(at["image"]); info = pd.read_csv(at["info"])
    lab = np.asarray(img.get_fdata()).astype(int); ctx = info[info.structure == "cortex"]
    idm = {int(r.id): (str(r.hemisphere), str(r.label).lower()) for r in ctx.itertuples()}
    vox = nib.affines.apply_affine(np.linalg.inv(img.affine), xyz); vi = np.rint(vox).astype(int)
    inb = np.all((vi >= 0) & (vi < np.array(lab.shape)), axis=1)
    pl = np.zeros(len(xyz), int); pl[inb] = lab[vi[inb, 0], vi[inb, 1], vi[inb, 2]]
    cvx = np.argwhere(np.isin(lab, list(idm))); need = np.where(pl == 0)[0]
    dd, jj = cKDTree(cvx).query(vox[need]); ok = dd <= 4; hit = cvx[jj[ok]]
    pl[need[ok]] = lab[hit[:, 0], hit[:, 1], hit[:, 2]]; pl[~np.isin(pl, list(idm))] = 0

    reg = {k: (pl == i) for i, k in idm.items() if (pl == i).sum() >= 8}
    keys = [k for k in reg]
    cov = np.array([np.nanmean(rc[reg[k]]) for k in keys])
    myv = np.array([np.nanmean(mye[reg[k]]) for k in keys])
    cen = np.array([xyz[reg[k]].mean(0) for k in keys])
    covpct = rankdata(cov) / len(keys) * 100.0
    myepct = rankdata(myv) / len(keys) * 100.0

    # spin perms over the 52 region centroids (valid geometry, L/R separate)
    c = cen - cen.mean(0); sph = c / np.linalg.norm(c, axis=1, keepdims=True)
    tree = cKDTree(sph); rng = np.random.default_rng(0)
    perms = [tree.query(sph @ _haar_rotation(rng).T)[1] for _ in range(2000)]

    def reach(refpct, burden):
        obs = float(np.average(refpct, weights=burden))
        null = np.array([np.average(refpct[p], weights=burden) for p in perms])  # spin the REFERENCE map
        p2 = float((np.sum(np.abs(null - 50) >= abs(obs - 50)) + 1) / (len(perms) + 1))
        return obs, p2, float(null.mean())

    out = {"_def": "reachability with SPIN null (rotate reference map); 52 hemi-regions; two-sided p on |obs-50|"}
    print(f"{'disorder':14s} {'recon':>18s}   {'myelin(hierarchy)':>20s}   burden")
    for pth in sorted(glob.glob(str(ROOT / "data_external/enigma/cortical_thickness_*.csv"))):
        nm = Path(pth).stem.replace("cortical_thickness_", ""); dm = enig(pth)
        idx = [i for i, k in enumerate(keys) if k in dm]
        d = np.array([dm[keys[i]] for i in idx]); burden = np.maximum(0.0, -d)
        if burden.sum() < 1e-9:
            continue
        # restrict maps to the disorder's regions, re-percentile within those
        cp = rankdata(cov[idx]) / len(idx) * 100.0; mp = rankdata(myv[idx]) / len(idx) * 100.0
        sub = cen[idx]; cc = sub - sub.mean(0); ss = cc / np.linalg.norm(cc, axis=1, keepdims=True)
        tr = cKDTree(ss); rng2 = np.random.default_rng(0)
        pm = [tr.query(ss @ _haar_rotation(rng2).T)[1] for _ in range(2000)]
        def reach2(refpct):
            obs = float(np.average(refpct, weights=burden))
            null = np.array([np.average(refpct[p], weights=burden) for p in pm])
            return obs, float((np.sum(np.abs(null - 50) >= abs(obs - 50)) + 1) / (len(pm) + 1))
        rc_o, rc_p = reach2(cp); my_o, my_p = reach2(mp)
        out[nm] = {"recon_reach": rc_o, "recon_spin_p": rc_p, "myelin_reach": my_o,
                   "myelin_spin_p": my_p, "burden": float(burden.sum()), "mean_d": float(d.mean()), "n": len(idx)}
        s1 = "*" if rc_p < 0.05 else " "; s2 = "*" if my_p < 0.05 else " "
        print(f"  {nm:12s} {rc_o:5.1f} (spin p={rc_p:.3f}){s1}   {my_o:5.1f} (spin p={my_p:.3f}){s2}   {burden.sum():5.2f}")
    out.update(pi_provenance())   # which coupling produced these numbers
    (ROOT / "outputs/logs/section6_reachability_spin.json").write_text(json.dumps(out, indent=2))
    print("wrote section6_reachability_spin.json")


if __name__ == "__main__":
    main()
