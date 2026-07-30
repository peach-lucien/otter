#!/usr/bin/env python3
"""Anchor-driven WARPED cross-species spatial cost vs the naive per-species Euclidean one.

The production M_xyz is Euclidean distance between PER-SPECIES-normalised coordinates. It is
naive: the human frontal pole normalises into a coordinate corner the mouse never reaches, so
human dorsolateral PFC (Yeo-17 Control B) is penalised by construction. This corrupts the
column-sum "coverage": coverage tracks spatial position (rho ~ -0.62) and the dlPFC deficit
vanishes when the xyz term is zeroed.

Anchor-driven warp: fit a smooth mouse->human coordinate map from the 42 Garin homolog anchor
pairs (thin-plate-spline RBF, and an affine baseline), warp ALL mouse coords into human space,
build M_xyz as warped-mouse<->human Euclidean distance, refit, and test whether coverage stops
being position-driven and whether the dlPFC deficit survives.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/section5_coverage_rigor"))

from homer.data import load_cached, load_pi                     # noqa: E402
from homer.data.anchors import get_anchor_index                 # noqa: E402
from homer.eval.nulls import _haar_rotation                     # noqa: E402
from homer.models import MultimodalFGW                          # noqa: E402
from beauchamp_scorer import BeauchampScorer                    # noqa: E402

N_SPIN = 2000
SEED = 0


def affine_warp(src, dst):
    A = np.hstack([src, np.ones((len(src), 1))])
    W, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return lambda pts: np.hstack([pts, np.ones((len(pts), 1))]) @ W


def tps_warp(src, dst):
    rbf = RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=1e-3)
    return lambda pts: rbf(pts)


def warped_M_xyz(warped_mouse, hxyz):
    sq_m = (warped_mouse ** 2).sum(1, keepdims=True)
    sq_h = (hxyz ** 2).sum(1, keepdims=True)
    d2 = sq_m + sq_h.T - 2.0 * warped_mouse @ hxyz.T
    d = np.sqrt(np.clip(d2, 0.0, None))
    return d / max(d.max(), 1e-9)


def coverage(pi):
    return np.log10(np.maximum(pi.sum(0), 1e-300))


def yeo17(nr):
    rows = [l.split("\t") for l in
            (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    nmap = {int(p[0]): p[1] for p in rows}
    return np.array([nmap.get(k, "?_?_?").split("_")[2] if k in nmap else "?" for k in nr])


def spin_perms(coords, n=N_SPIN, seed=SEED):
    c = coords - coords.mean(0)
    sph = c / np.linalg.norm(c, axis=1, keepdims=True)
    tree = cKDTree(sph)
    rng = np.random.default_rng(seed)
    return [tree.query(sph @ _haar_rotation(rng).T)[1] for _ in range(n)]


def block_gap(sig, sel, perms):
    f = lambda s: s[sel].mean() - s[~sel].mean()               # noqa: E731
    obs = f(sig)
    null = np.abs([f(sig[p]) for p in perms])
    return {"gap_sd": float(obs),
            "spin_p": float((np.sum(null >= abs(obs)) + 1) / (len(perms) + 1))}


def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    mxyz = M.var[["x", "y", "z"]].to_numpy(float)
    hxyz = H.var[["x", "y", "z"]].to_numpy(float)

    im = get_anchor_index(M.var)
    ih = get_anchor_index(H.var)
    hpos = {int(p): k for k, p in zip(ih.pos, ih.pair_ids)}
    pairs = [(int(mp), hpos[int(pid)]) for mp, pid in zip(im.pos, im.pair_ids) if int(pid) in hpos]
    src = mxyz[[a for a, b in pairs]]
    dst = hxyz[[b for a, b in pairs]]
    print(f"[anchors] {len(pairs)} mouse->human control-point pairs")

    fa = affine_warp(src, dst)
    ft = tps_warp(src, dst)
    warped_affine = fa(mxyz)
    warped_tps = ft(mxyz)
    aff_res = float(np.sqrt(((fa(src) - dst) ** 2).sum(1)).mean())
    tps_res = float(np.sqrt(((ft(src) - dst) ** 2).sum(1)).mean())
    print(f"[warp] mean anchor residual  affine={aff_res:.3f}  tps={tps_res:.3f}")

    Mxyz_affine = warped_M_xyz(warped_affine, hxyz)
    Mxyz_tps = warped_M_xyz(warped_tps, hxyz)

    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")

    def fit(M_xyz):
        m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7, epsilon=5e-3,
                          xyz_weight=0.5, lam_anchor=1.0, alpha=0.5)
        m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"], M_xyz=M_xyz, region_anchors=[])
        return m.pi.astype(np.float64)

    print("[fit] euclid baseline ..."); pi_euclid = fit(costs["M_xyz"])
    print("[fit] affine warp ...");    pi_affine = fit(Mxyz_affine)
    print("[fit] tps warp ...");       pi_tps = fit(Mxyz_tps)
    np.save("/var/tmp/pi_warp_euclid_baseline.npy", pi_euclid)
    np.save("/var/tmp/pi_warp_affine.npy", pi_affine)
    np.save("/var/tmp/pi_warp_tps.npy", pi_tps)

    pi_prod = load_pi().astype(np.float64)

    sc = BeauchampScorer()
    couplings = {"production": pi_prod, "euclid_baseline": pi_euclid,
                 "warp_affine": pi_affine, "warp_tps": pi_tps}
    beau, per_pair = {}, {}
    for name, pi in couplings.items():
        res = sc.score(pi)
        agg = res["__aggregate__"]
        beau[name] = {k: float(agg[k]) for k in
                      ("top1", "top5", "mean_mass_in_region", "enrichment_top1", "n_pairs")}
        per_pair[name] = {p: float(res[p]["top1"]) for p in res if p != "__aggregate__"}
        print(f"[beauchamp] {name:<16} top1={agg['top1']:.3f} top5={agg['top5']:.3f} "
              f"mean_mass={agg['mean_mass_in_region']:.3f} enr={agg['enrichment_top1']:.2f}")

    pairnames = list(per_pair["euclid_baseline"].keys())
    deltas_tps = sorted(((p, per_pair["warp_tps"][p] - per_pair["euclid_baseline"][p]) for p in pairnames),
                        key=lambda kv: kv[1], reverse=True)
    deltas_aff = sorted(((p, per_pair["warp_affine"][p] - per_pair["euclid_baseline"][p]) for p in pairnames),
                        key=lambda kv: kv[1], reverse=True)

    mye = np.asarray(json.loads(
        (ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())["myelin_per_parcel"], float)
    nr = np.asarray(json.loads(
        (ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    net = yeo17(nr)

    def spatial_analysis(pi, M_xyz):
        cov = coverage(pi)
        spatial_iso = M_xyz.min(0)
        m = np.isfinite(cov) & np.isfinite(mye)
        rho, pval = spearmanr(cov[m], spatial_iso[m])
        z = (cov[m] - cov[m].mean()) / cov[m].std()
        sel = net[m] == "ContB"
        gap = block_gap(z, sel, spin_perms(hxyz[m]))
        return {"spearman_coverage_vs_spatial_isolation": float(rho),
                "spearman_p": float(pval),
                "n_cortical_parcels": int(m.sum()),
                "contB_deficit_sd": gap["gap_sd"],
                "contB_spin_p": gap["spin_p"],
                "contB_mean_coverage_sd": float(z[sel].mean())}

    spatial = {
        "production": spatial_analysis(pi_prod, costs["M_xyz"]),
        "euclid_baseline": spatial_analysis(pi_euclid, costs["M_xyz"]),
        "warp_affine": spatial_analysis(pi_affine, Mxyz_affine),
        "warp_tps": spatial_analysis(pi_tps, Mxyz_tps),
    }
    for name, s in spatial.items():
        print(f"[spatial] {name:<16} rho(cov,iso)={s['spearman_coverage_vs_spatial_isolation']:+.3f} "
              f"ContB deficit={s['contB_deficit_sd']:+.2f} SD spin p={s['contB_spin_p']:.4f}")

    out = {
        "_finding": "anchor-driven warp vs naive per-species Euclidean spatial cost.",
        "config": {"base": "use_sc=True sc=0.3 fc=0.7 eps=5e-3 xyz_weight=0.5 lam_anchor=1.0 alpha=0.5",
                   "n_anchor_pairs": len(pairs),
                   "warp_mean_anchor_residual": {"affine": aff_res, "tps": tps_res},
                   "n_spin": N_SPIN},
        "beauchamp": beau,
        "beauchamp_per_pair_top1": per_pair,
        "top_improvements_tps_vs_euclid": [[p, float(d)] for p, d in deltas_tps[:8]],
        "top_regressions_tps_vs_euclid": [[p, float(d)] for p, d in deltas_tps[-5:]],
        "top_improvements_affine_vs_euclid": [[p, float(d)] for p, d in deltas_aff[:8]],
        "spatial": spatial,
        "reference": {"production_beauchamp_top1": 0.457,
                      "production_rho_cov_iso": -0.62,
                      "production_contB_deficit_sd": -1.20,
                      "production_contB_spin_p": 0.0005},
    }
    outpath = ROOT / "outputs/logs/section5_warp_prototype.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {outpath}")


if __name__ == "__main__":
    main()
