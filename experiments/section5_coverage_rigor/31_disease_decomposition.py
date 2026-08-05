"""Decompose disorder cortical thinning into a HIERARCHY component (myelin/S-A axis, conserved,
transfers to mouse) and an EXPANSION component (reconstruction-coverage, human-specific, does not).

Per disorder, 52 hemi-regions: partial Spearman of thinning d against each axis controlling the
other, with a SPIN null (rotate the predictor map; valid geometry, L/R separate). If disease loads
only on hierarchy (d~myelin|coverage significant, d~coverage|myelin null), disease vulnerability is
the conserved component; any residual d~coverage|myelin would be a human-specific/un-transferable part.

Writes outputs/logs/section6_decomposition.json
"""
from __future__ import annotations
import csv, glob, json, sys
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
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


def presid(y, z):
    """residual of ranked y after removing ranked z (linear)."""
    A = np.c_[np.ones_like(z), z]
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


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
    print(f"axis check: Spearman(recon-coverage, myelin) over {len(keys)} regions = "
          f"{spearmanr(cov, myv).statistic:+.3f}  (near 0 => distinct axes)")

    out = {"_note": "partial Spearman of thinning d vs each axis, controlling the other; spin null on the predictor"}
    print(f"\n{'disorder':13s} {'d~cov|myelin':>16s} {'d~myelin|cov':>16s}")
    for pth in sorted(glob.glob(str(ROOT / "data_external/enigma/cortical_thickness_*.csv"))):
        nm = Path(pth).stem.replace("cortical_thickness_", ""); dm = enig(pth)
        idx = [i for i, k in enumerate(keys) if k in dm]
        d = rankdata([dm[keys[i]] for i in idx])
        cv = rankdata(cov[idx]); my = rankdata(myv[idx]); C = cen[idx]
        # partials
        d_c = presid(d, my); cv_c = presid(cv, my)      # control myelin
        d_m = presid(d, cv); my_c = presid(my, cv)      # control coverage
        r_cov = spearmanr(d_c, cv_c).statistic
        r_mye = spearmanr(d_m, my_c).statistic
        # spin the predictor (coverage / myelin) over region centroids
        cc = C - C.mean(0); ss = cc / np.linalg.norm(cc, axis=1, keepdims=True)
        tr = cKDTree(ss); rng = np.random.default_rng(0)
        perms = [tr.query(ss @ _haar_rotation(rng).T)[1] for _ in range(2000)]
        null_cov = [spearmanr(presid(d, my), presid(cv[p], my)).statistic for p in perms]
        null_mye = [spearmanr(presid(d, cv), presid(my[p], cv)).statistic for p in perms]
        p_cov = (np.sum(np.abs(null_cov) >= abs(r_cov)) + 1) / (len(perms) + 1)
        p_mye = (np.sum(np.abs(null_mye) >= abs(r_mye)) + 1) / (len(perms) + 1)
        out[nm] = {"partial_cov": float(r_cov), "partial_cov_spin_p": float(p_cov),
                   "partial_mye": float(r_mye), "partial_mye_spin_p": float(p_mye), "n": len(idx)}
        s1 = "*" if p_cov < 0.05 else " "; s2 = "*" if p_mye < 0.05 else " "
        print(f"  {nm:11s} {r_cov:+.2f} (p={p_cov:.3f}){s1}   {r_mye:+.2f} (p={p_mye:.3f}){s2}")
    out.update(pi_provenance())   # which coupling produced these numbers
    (ROOT / "outputs/logs/section6_decomposition.json").write_text(json.dumps(out, indent=2))
    print("wrote section6_decomposition.json")


if __name__ == "__main__":
    main()
