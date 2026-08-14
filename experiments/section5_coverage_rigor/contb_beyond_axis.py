"""Is the Control B deficit more than its position on the sensorimotor-association axis predicts?

Run: cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/contb_beyond_axis.py

The tertile split and the network sweep are not independent. If the network ordering is just the
sensorimotor-association axis reappearing at network level, then naming Control B adds a label and
nothing else. This asks whether Control B sits below what its axis position alone would give.

Reconstruction accuracy is residualised on the Sydnor 2021 axis at parcel level, and the Control B
block gap is recomputed on the residual with a spin null that re-residualises on every rotation.
Machinery matches 11_dlpfc_deficit.py.
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

PI = "outputs/coupling/pi_canonical.npy"

from otter.data import load_cached                     # noqa: E402
from otter.eval.nulls import _haar_rotation            # noqa: E402

N_SPIN, SEED, MIN_N = 2000, 0, 30


def spin_perms(coords, n=N_SPIN, seed=SEED):
    c = coords - coords.mean(0)
    sph = c / np.linalg.norm(c, axis=1, keepdims=True)
    tree = cKDTree(sph)
    rng = np.random.default_rng(seed)
    return [tree.query(sph @ _haar_rotation(rng).T)[1] for _ in range(n)]


def resid_on(y, x):
    """Rank-residual of y after removing x."""
    ry, rx = rankdata(y), rankdata(x)
    X = np.c_[np.ones_like(rx), rx]
    return ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]


def main() -> int:
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    Mfc = np.asarray(M.uns["fc_mean"], float)
    Hfc = np.asarray(H.uns["fc_mean"], float)
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    pi = np.load(ROOT / PI)
    sha = hashlib.sha256((ROOT / PI).read_bytes()).hexdigest()
    pit = pi / np.maximum(pi.sum(0), 1e-300)
    pred = pit.T @ Mfc @ pit
    rc = np.full(pred.shape[0], np.nan)
    for j in range(pred.shape[0]):
        a, b = pred[j].copy(), Hfc[j].copy()
        a[j] = np.nan; b[j] = np.nan
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10 and a[ok].std() > 1e-9:
            rc[j] = np.corrcoef(a[ok], b[ok])[0, 1]

    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    rows = [l.split("\t") for l in (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    lut = {int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}
    net = np.array([lut.get(int(k), "?") for k in nr])
    batt = json.loads((ROOT / "data_external/published_cortical_maps.json").read_text())["maps"]
    v = batt["Sydnor2021 S–A axis"]
    sa_by_region = dict(zip(np.asarray(v["schaefer_ids"], int), np.asarray(v["map_values"], float)))
    sa = np.array([sa_by_region.get(int(k), np.nan) for k in nr])

    m = np.isfinite(rc) & np.isfinite(sa)
    print(f"{m.sum()} parcels carry both reconstruction accuracy and the Sydnor axis")

    # ---- does the network ordering recapitulate the axis ----
    nets = sorted({u for u in set(net[m]) if (net[m] == u).sum() >= MIN_N})
    net_acc = np.array([np.nanmean(rc[m & (net == u)]) for u in nets])
    net_sa = np.array([np.nanmean(sa[m & (net == u)]) for u in nets])
    rho_net = spearmanr(net_acc, net_sa).statistic
    print(f"\nacross the {len(nets)} networks, mean accuracy against mean axis position: "
          f"rho = {rho_net:+.3f}")
    order = np.argsort(net_acc)
    for i in order:
        print(f"   {nets[i]:14s} accuracy {net_acc[i]:.3f}   axis {net_sa[i]:+.2f}")

    # ---- is Control B below what its axis position predicts ----
    z = (rc[m] - rc[m].mean()) / rc[m].std()
    sel = net[m] == "ContB"
    perms = spin_perms(xyz[m])
    raw_gap = float(z[sel].mean() - z[~sel].mean())

    r_acc = resid_on(rc[m], sa[m])
    r_acc = (r_acc - r_acc.mean()) / r_acc.std()
    res_gap = float(r_acc[sel].mean() - r_acc[~sel].mean())
    null = []
    for p in perms:
        rp = resid_on(rc[m][p], sa[m])
        rp = (rp - rp.mean()) / rp.std()
        null.append(rp[sel].mean() - rp[~sel].mean())
    null = np.abs(np.asarray(null))
    res_p = float((np.sum(null >= abs(res_gap)) + 1) / (len(perms) + 1))
    print(f"\nControl B gap, raw                       {raw_gap:+.3f} SD")
    print(f"Control B gap, axis position removed     {res_gap:+.3f} SD   spin p = {res_p:.4f}")

    out = {"pi_file": PI, "pi_sha256": sha,
           "n_parcels": int(m.sum()), "n_spin": N_SPIN,
           "network_ordering_vs_axis_rho": round(float(rho_net), 3),
           "contb_gap_raw_sd": round(raw_gap, 3),
           "contb_gap_axis_removed_sd": round(res_gap, 3),
           "contb_gap_axis_removed_spin_p": round(res_p, 4),
           "network_means": {u: {"accuracy": round(float(a), 4), "axis": round(float(s), 3)}
                             for u, a, s in zip(nets, net_acc, net_sa)}}
    (ROOT / "outputs/logs/contb_beyond_axis.json").write_text(json.dumps(out, indent=1))
    print("\nwrote contb_beyond_axis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
