"""Disorder-level 'mouse-reachability': how much of each disorder's cortical thinning burden lands
on cortex the mouse CAN reconstruct, vs cortex it cannot.

A per-disorder descriptive statistic, robust to near-null disease maps. For each disorder:
  burden(region) = max(0, -d)                      # thinning magnitude, 0 if thickening
  reachability   = burden-weighted mean of the reconstruction-coverage PERCENTILE across regions
A disorder whose burden is spatially uniform scores ~50. Concentrated in LOW-reconstruction (mouse-
un-reachable) cortex -> <50. Near-null disorders (ASD) have ~0 burden everywhere -> ~50, no false
signal. Null: permute burden across regions (10000x). One-sided p = fraction of null <= observed
(is the disorder's burden significantly biased toward un-reachable cortex?).

52 DK hemi-regions (L/R separate, valid geometry), reconstruction-coverage = FC+SC on pi_canonical.
Writes outputs/logs/section6_reachability.json
"""
from __future__ import annotations
import csv, glob, json, sys
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance  # noqa: E402
import abagen                                            # noqa: E402


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
    cov = {k: float(np.nanmean(rc[m])) for k, m in reg.items()}
    keys = [k for k in cov if np.isfinite(cov[k])]
    covpct = dict(zip(keys, rankdata([cov[k] for k in keys]) / len(keys) * 100.0))

    rng = np.random.default_rng(0); NPERM = 10000
    out = {"_def": "reachability = thinning-burden-weighted mean reconstruction-coverage percentile; "
                   "<50 = burden in mouse-un-reachable cortex. FC+SC recon on pi_canonical, 52 hemi-regions."}
    rows = []
    for p in sorted(glob.glob(str(ROOT / "data_external/enigma/cortical_thickness_*.csv"))):
        nm = Path(p).stem.replace("cortical_thickness_", ""); dm = enig(p)
        ks = [k for k in keys if k in dm]
        c = np.array([covpct[k] for k in ks]); d = np.array([dm[k] for k in ks])
        burden = np.maximum(0.0, -d)
        if burden.sum() < 1e-9:
            out[nm] = {"reachability": None, "note": "no thinning burden"}; continue
        obs = float(np.average(c, weights=burden))
        null = np.array([np.average(c, weights=burden[rng.permutation(len(burden))]) for _ in range(NPERM)])
        p_lo = float((np.sum(null <= obs) + 1) / (NPERM + 1))
        out[nm] = {"reachability": obs, "null_mean": float(null.mean()), "p_less_reachable": p_lo,
                   "total_burden": float(burden.sum()), "mean_d": float(d.mean()), "n": len(ks)}
        rows.append((nm, obs, p_lo, float(burden.sum()), float(d.mean())))
    for nm, obs, p_lo, b, md in sorted(rows, key=lambda r: r[1]):
        star = "***" if p_lo < 0.05 else ""
        print(f"  {nm:14s} reachability={obs:5.1f} (null~50)  p={p_lo:.3f}{star}  burden={b:5.2f}  meanD={md:+.2f}")
    out.update(pi_provenance())   # which coupling produced these numbers
    (ROOT / "outputs/logs/section6_reachability.json").write_text(json.dumps(out, indent=2))
    print("wrote section6_reachability.json")


if __name__ == "__main__":
    main()
