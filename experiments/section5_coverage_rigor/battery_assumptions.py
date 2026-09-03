"""Evaluate the interpretation and assumptions of the cortical-map battery.

Run: cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/battery_assumptions.py

Recomputes parcel-wise reconstruction accuracy, then evaluates four questions:

  1. How independent are the seven maps? Inter-correlation and a principal component.
  2. Does any map survive once the shared axis is partialled out?
  3. Is the effect explained by where the supervision is, rather than by evolution?
  4. Is it explained by how well each parcel's own functional connectivity is estimated?

The residual and partial-correlation checks are distinct from the primary spatial
tests. Where required, they use the corresponding spatial rotation procedure.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, rankdata

# experiments/section5_coverage_rigor/ -> parents[2] is the package dir, as elsewhere in the repo.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached                     # noqa: E402
from otter.eval.nulls import _haar_rotation            # noqa: E402

MAPS = [
    ("Xu2020 macaque→human expansion", "Xu2020 expansion", "evolution"),
    ("Hill2010 macaque→human expansion", "Hill2010 expansion", "evolution"),
    ("Hill2010 developmental expansion", "Hill2010 development", "evolution"),
    ("Xu2020 macaque–human FC homology", "Xu2020 FC homology", "evolution"),
    ("Sydnor2021 S–A axis", "Sydnor2021 S-A", "hierarchy"),
    ("Margulies2016 principal gradient", "Margulies gradient", "hierarchy"),
    ("HCP T1w/T2w hierarchy", "HCP T1w/T2w", "hierarchy"),
]


PI = "outputs/coupling/pi_canonical.npy"


def pi_stamp():
    """Record which coupling produced the log, at the write site."""
    h = hashlib.sha256((ROOT / PI).read_bytes()).hexdigest()
    return {"pi_file": PI, "pi_sha256": h}


def reconstruction_accuracy():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    Mfc = np.asarray(M.uns["fc_mean"], float)
    Hfc = np.asarray(H.uns["fc_mean"], float)
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    pi = np.load(ROOT / PI)
    ph = pi.sum(0)
    pit = pi / np.maximum(ph, 1e-300)
    pred = pit.T @ Mfc @ pit
    n = pred.shape[0]
    rc = np.full(n, np.nan)
    for j in range(n):
        a = pred[j].copy(); b = Hfc[j].copy()
        a[j] = np.nan; b[j] = np.nan
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10 and a[ok].std() > 1e-9:
            rc[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return rc, xyz, Hfc, ph, M, H


def spin_p(cc, mv, cen, rho, N=1000, seed=0):
    c = cen - cen.mean(0)
    s = c / np.linalg.norm(c, axis=1, keepdims=True)
    t = cKDTree(s)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(N):
        idx = t.query(s @ _haar_rotation(rng).T)[1]
        null.append(spearmanr(cc[idx], mv).statistic)
    null = np.abs(np.asarray(null))
    return float((np.sum(null >= abs(rho)) + 1) / (N + 1)), null


def spin_p_resid(cc, mv, cen, ctrl, rho, N=1000, seed=0):
    """Spin null for a residual statistic: rotate the accuracy map, then re-residualise."""
    c = cen - cen.mean(0)
    s = c / np.linalg.norm(c, axis=1, keepdims=True)
    t = cKDTree(s)
    rng = np.random.default_rng(seed)
    rm = rankdata(mv); rk = rankdata(ctrl)
    Xm = np.c_[np.ones_like(rm), rk]
    mv_r = rm - Xm @ np.linalg.lstsq(Xm, rm, rcond=None)[0]
    null = []
    for _ in range(N):
        idx = t.query(s @ _haar_rotation(rng).T)[1]
        rcp = rankdata(cc[idx])
        cc_r = rcp - Xm @ np.linalg.lstsq(Xm, rcp, rcond=None)[0]
        null.append(spearmanr(cc_r, mv_r).statistic)
    null = np.abs(np.asarray(null))
    return float((np.sum(null >= abs(rho)) + 1) / (N + 1))


def partial_spearman(a, b, c):
    """Spearman of a and b with c partialled out, on ranks."""
    ra, rb, rc_ = rankdata(a), rankdata(b), rankdata(c)
    X = np.c_[np.ones_like(rc_), rc_]
    ea = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    eb = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    return float(spearmanr(ea, eb).statistic)


def main() -> int:
    rc, xyz, Hfc, ph, M, H = reconstruction_accuracy()
    batt = json.loads((ROOT / "data_external/published_cortical_maps.json").read_text())["maps"]
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)

    # region series, built the way the figure builds them
    series = {}
    for key, short, grp in MAPS:
        v = batt.get(key, {})
        if "schaefer_ids" not in v:
            print(f"missing {key}")
            continue
        mp = dict(zip(np.asarray(v["schaefer_ids"], int), np.asarray(v["map_values"], float)))
        ids = [k for k in range(1, 401) if (nr == k).any() and k in mp]
        cc = np.array([np.nanmean(rc[nr == k]) for k in ids])
        mv = np.array([mp[k] for k in ids])
        cen = np.array([xyz[nr == k].mean(0) for k in ids])
        series[short] = dict(ids=ids, cc=cc, mv=mv, cen=cen, grp=grp)

    shorts = list(series)
    # a common id set so the maps can be compared to one another
    common = sorted(set.intersection(*[set(series[s]["ids"]) for s in shorts]))
    print(f"{len(common)} Schaefer regions defined on all seven maps\n")
    pos = {s: {k: i for i, k in enumerate(series[s]["ids"])} for s in shorts}
    Mmat = np.column_stack([series[s]["mv"][[pos[s][k] for k in common]] for s in shorts])
    acc = series[shorts[0]]["cc"][[pos[shorts[0]][k] for k in common]]
    cen = series[shorts[0]]["cen"][[pos[shorts[0]][k] for k in common]]

    out = {"n_common_regions": len(common)}

    # ---- 1. how independent are the maps ----
    R = spearmanr(Mmat).statistic
    print("Spearman between the seven maps")
    hdr = "".join(f"{s[:12]:>14s}" for s in shorts)
    print(f"{'':26s}{hdr}")
    for i, s in enumerate(shorts):
        print(f"{s:26s}" + "".join(f"{R[i, j]:14.2f}" for j in range(len(shorts))))
    off = R[np.triu_indices(len(shorts), 1)]
    print(f"\nmean |rho| off diagonal {np.abs(off).mean():.2f}   "
          f"max {np.abs(off).max():.2f}   min {np.abs(off).min():.2f}")
    out["intermap_spearman"] = {shorts[i]: {shorts[j]: round(float(R[i, j]), 3)
                                            for j in range(len(shorts))} for i in range(len(shorts))}
    out["intermap_abs_mean"] = round(float(np.abs(off).mean()), 3)

    # sign-aligned PCA on ranks, so one axis is measured in one direction
    Z = np.column_stack([rankdata(Mmat[:, i]) for i in range(Mmat.shape[1])])
    Z = (Z - Z.mean(0)) / Z.std(0)
    sgn = np.sign(spearmanr(Z, np.asarray(acc)).statistic[:-1, -1])
    sgn[sgn == 0] = 1
    Zs = Z * sgn
    U, S, Vt = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    pc1 = U[:, 0] * S[0]
    if spearmanr(pc1, Zs.mean(1)).statistic < 0:
        pc1 = -pc1
    print(f"\nvariance explained by each component {np.round(ev, 3)}")
    print(f"PC1 carries {ev[0] * 100:.0f} per cent of the variance across the seven maps")
    out["pca_explained"] = [round(float(x), 4) for x in ev]

    rho_pc1 = spearmanr(acc, pc1).statistic
    p_pc1, _ = spin_p(np.asarray(acc), pc1, cen, rho_pc1)
    print(f"reconstruction accuracy vs PC1   rho = {rho_pc1:+.3f}   spin p = {p_pc1:.4f}")
    out["accuracy_vs_pc1"] = {"rho": round(float(rho_pc1), 3), "spin_p": round(p_pc1, 4)}

    # ---- 2. does anything survive the shared axis ----
    print("\nEach map after partialling out PC1 of the other six")
    out["partial"] = {}
    for i, s in enumerate(shorts):
        others = np.delete(Zs, i, axis=1)
        Uo, So, _ = np.linalg.svd(others - others.mean(0), full_matrices=False)
        pco = Uo[:, 0] * So[0]
        raw = spearmanr(acc, Mmat[:, i]).statistic
        par = partial_spearman(np.asarray(acc), Mmat[:, i], pco)
        pp = spin_p_resid(np.asarray(acc), Mmat[:, i], cen, pco, par)
        print(f"  {s:26s} raw {raw:+.3f}   partial {par:+.3f}   spin p {pp:.3f}")
        out["partial"][s] = {"raw_rho": round(float(raw), 3),
                             "partial_rho": round(float(par), 3), "spin_p": round(pp, 4)}

    # ---- 3. supervision density ----
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    print(f"\nfull_costs keys {list(costs.keys())}")

    # ---- 4. is accuracy driven by how well each parcel's own FC is estimated ----
    # A parcel whose measured FC row is weak or unstructured cannot be reconstructed by anything.
    fcsd = np.nanstd(Hfc, axis=1)
    fcabs = np.nanmean(np.abs(Hfc), axis=1)
    reg_sd = np.array([np.nanmean(fcsd[nr == k]) for k in common])
    reg_abs = np.array([np.nanmean(fcabs[nr == k]) for k in common])
    reg_mass = np.array([np.nanmean(np.log10(np.maximum(ph, 1e-300))[nr == k]) for k in common])
    for nm, ctrl in (("FC row dispersion", reg_sd), ("mean |FC|", reg_abs),
                     ("log10 transported mass", reg_mass)):
        r0 = spearmanr(acc, ctrl).statistic
        print(f"\naccuracy vs {nm}: rho = {r0:+.3f}")
        out.setdefault("controls", {})[nm] = {"rho_vs_accuracy": round(float(r0), 3), "maps": {}}
        for i, s in enumerate(shorts):
            raw = spearmanr(acc, Mmat[:, i]).statistic
            par = partial_spearman(np.asarray(acc), Mmat[:, i], ctrl)
            print(f"    {s:26s} raw {raw:+.3f}  partial {par:+.3f}")
            out["controls"][nm]["maps"][s] = {"raw": round(float(raw), 3),
                                              "partial": round(float(par), 3)}

    out.update(pi_stamp())
    (ROOT / "outputs/logs/battery_assumptions.json").write_text(json.dumps(out, indent=1))
    print("\nwrote battery_assumptions.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
