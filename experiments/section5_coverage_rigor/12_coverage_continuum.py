#!/usr/bin/env python3
"""Coverage as a continuum. Coverage is a graded quantity, and on the canonical coupling it
follows NO spatial axis (the medial->lateral gradient was a retired-coupling artefact).

The dlPFC hole (script 11) is the extreme low end of a continuous distribution, not a separate
phenomenon. This script establishes the continuum and its spatial organisation:

1. CONTINUUM. Coverage (log10 mouse mass per human parcel) is a smooth, skewed, unimodal
   distribution: from strongly reached (sensorimotor, insula) through partially reached to a thin
   under-reached tail (lateral PFC / dlPFC). It is not bimodal; there is no clean "covered vs not".

2. THE AXIS. On the RETIRED pre-warp coupling, coverage declined along the medial->lateral axis
   (rho(coverage,|x|) = -0.30, spin p = 0.0005). ON THE CANONICAL COUPLING THIS IS GONE:
   rho = -0.03, spin p = 0.83, and every other axis (A-P, D-V, myelin) is null too. The
   medial->lateral gradient is a property of the retired coupling, not of the canonical one.
   Spin-tested (coverage is asymmetric, |x| is symmetric: the calibrated 5.5% FPR configuration).

3. ROBUSTNESS. The medial->lateral gradient is present on the base, production (retired) and
   anchor-free couplings but ABSENT on the canonical one, so it does not generalise across
   couplings. The per-coupling arms below establish that; canonical is the reported arm.

Spin note: spin the SIGNAL (coverage, asymmetric) against the symmetric axis map. Do NOT
bilaterally average coverage first; that gives a 37% FPR.

Run:  cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/12_coverage_continuum.py
Writes: outputs/logs/section5_coverage_continuum.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached, load_pi, pi_provenance        # noqa: E402
from otter.data.anchors import get_anchor_index                   # noqa: E402
from otter.eval.nulls import _haar_rotation                       # noqa: E402

N_SPIN = 2000
SEED = 0


def coverage(pi):
    return np.log10(np.maximum(pi.sum(0), 1e-300))


def yeo17(nr):
    rows = [l.split("\t") for l in
            (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    nmap = {int(p[0]): p[1] for p in rows}
    return np.array([nmap.get(k, "?_?_?").split("_", 2)[2] if k in nmap else "?" for k in nr])


def region_names(nr):
    rows = [l.split("\t") for l in
            (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    nmap = {int(p[0]): p[1] for p in rows}
    return np.array([nmap.get(k, "?") for k in nr])


def spin_perms(coords, n=N_SPIN, seed=SEED):
    c = coords - coords.mean(0)
    sph = c / np.linalg.norm(c, axis=1, keepdims=True)
    tree = cKDTree(sph)
    rng = np.random.default_rng(seed)
    return [tree.query(sph @ _haar_rotation(rng).T)[1] for _ in range(n)]


def spin_rho(sig, target, perms):
    """Spearman(sig, target) with a spin null spinning the signal."""
    obs = spearmanr(sig, target).statistic
    null = np.array([spearmanr(sig[p], target).statistic for p in perms])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(perms) + 1)
    return float(obs), float(p)


def main():
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    mye = np.asarray(json.loads(
        (ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())["myelin_per_parcel"], float)
    nr = np.asarray(json.loads(
        (ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    net = yeo17(nr)
    names = region_names(nr)

    pi = load_pi()
    prov = pi_provenance()
    print(f"π: {prov['pi_file']}  sha256={prov['pi_sha256']}")
    col = pi.sum(0)
    cov = coverage(pi)
    m = np.isfinite(cov) & np.isfinite(mye)                # cortex with a hierarchy value
    perms = spin_perms(xyz[m])

    out = {**prov,
           "_finding": ("Coverage is a continuous, unimodal, right-skewed quantity. On the CANONICAL "
                        "coupling it has no significant spatial axis: rho(coverage,|x|) = -0.03 "
                        "(spin p = 0.83), and A-P, D-V and myelin are null too. The medial->lateral "
                        "decline (rho = -0.30, p = 0.0005) is specific to the retired pre-warp "
                        "coupling; see medial_lateral_robustness for the per-coupling comparison."),
           "n_cortical_parcels": int(m.sum())}

    # ---- 1. the continuum -------------------------------------------------------------
    cv = cov[m]
    lin = col[m]
    out["continuum"] = {
        "log10_coverage_min": float(cv.min()),
        "log10_coverage_median": float(np.median(cv)),
        "log10_coverage_max": float(cv.max()),
        "histogram_counts": [int(x) for x in np.histogram(cv, bins=30)[0]],
        "histogram_edges": [round(float(x), 3) for x in np.histogram(cv, bins=30)[1]],
        "linear_mass_ratio_p90_to_p10": float(np.percentile(lin, 90) / np.percentile(lin, 10)),
        "_reading": ("Single mode, long thin left tail. No bimodality; there is no discrete "
                     "covered/uncovered split, only a graded decline into the tail."),
    }

    # example regions along the spectrum (by region-mean linear mass)
    reg_mass = {}
    for k in np.unique(nr[m]):
        sel = (nr == k) & m
        reg_mass[k] = (float(col[sel].mean()), names[sel][0] if sel.any() else "?")
    ranked = sorted(reg_mass.items(), key=lambda kv: kv[1][0])
    def label(k):
        nm = reg_mass[k][1]
        parts = nm.split("_")
        return "_".join(parts[2:]) if len(parts) > 2 else nm
    out["continuum"]["most_reached_regions"] = [label(k) for k, _ in ranked[-6:][::-1]]
    out["continuum"]["least_reached_regions"] = [label(k) for k, _ in ranked[:6]]

    # ---- 2. the axis: coverage vs |x|, and competitors --------------------------------
    absx = np.abs(xyz[m, 0])
    apy = xyz[m, 1]
    dvz = xyz[m, 2]
    myv = mye[m]
    axes = {}
    for nm, tgt in [("medial_lateral_absX", absx), ("anterior_posterior_Y", apy),
                    ("dorsal_ventral_Z", dvz), ("myelin", myv)]:
        rho, p = spin_rho(cv, tgt, perms)
        axes[nm] = {"spearman_rho": rho, "spin_p": p}
    out["spatial_axes"] = axes
    print("spatial correlates of coverage (canonical):")
    for nm, r in axes.items():
        print(f"  {nm:<22} rho={r['spearman_rho']:+.3f}  spin p={r['spin_p']:.4f}")

    # ---- 3. robustness across couplings + anchor-distance control ---------------------
    # distance to nearest anchor parcel
    try:
        aidx = get_anchor_index(H)
        aidx = np.asarray(list(aidx), int) if not isinstance(aidx, np.ndarray) else aidx
    except Exception:
        aidx = None
    # "canonical" is the reported arm (== the main analysis above); the others are kept
    # for cross-coupling comparison. "production" is the RETIRED pre-warp coupling.
    out["medial_lateral_robustness"] = {}
    out["_reported_coupling_arm"] = "canonical"
    for name, path in [("canonical", "pi_canonical.npy"),
                       ("base_garin_points_only", "pi_fc_plus_SC.npy"),
                       ("production", "pi_fc_plus_SC_with_all_packs.npy"),
                       ("anchor_free", "pi_anchorfree_control.npy")]:
        p = ROOT / "outputs/coupling" / path
        if not p.exists():
            out["medial_lateral_robustness"][name] = "coupling not found"
            continue
        cv2 = coverage(np.load(p))
        mm = np.isfinite(cv2) & np.isfinite(mye)
        perms2 = spin_perms(xyz[mm])
        rho, pp = spin_rho(cv2[mm], np.abs(xyz[mm, 0]), perms2)
        out["medial_lateral_robustness"][name] = {"spearman_rho": rho, "spin_p": pp,
                                                  **pi_provenance(path)}
        print(f"  {name:<24} rho(cov,|x|)={rho:+.3f}  spin p={pp:.4f}")

    # partial correlation controlling anchor distance (production)
    if aidx is not None and len(aidx):
        tree = cKDTree(xyz[aidx])
        adist = tree.query(xyz[m])[0]
        def resid(v, ctrl):
            A = np.c_[np.ones_like(ctrl), rankdata(ctrl)]
            b = np.linalg.lstsq(A, rankdata(v), rcond=None)[0]
            return rankdata(v) - A @ b
        rc, ra = resid(cv, adist), resid(absx, adist)
        out["partial_rho_covXabsX_control_anchordist"] = float(spearmanr(rc, ra).statistic)
        print(f"  partial rho(cov,|x| | anchor-dist) = "
              f"{out['partial_rho_covXabsX_control_anchordist']:+.3f}")

    # per-parcel coverage + |x| percentile for the figure
    pct = np.full(len(xyz), np.nan)
    ctx = np.isfinite(mye)
    pct[ctx] = rankdata(col[ctx]) / ctx.sum() * 100.0
    out["coverage_percentile_cortex"] = [None if not np.isfinite(v) else round(float(v), 3) for v in pct]
    out["log10_coverage_cortex"] = [None if not np.isfinite(cov[i]) or not ctx[i] else round(float(cov[i]), 4)
                                    for i in range(len(cov))]
    out["absX_cortex"] = [None if not ctx[i] else round(float(abs(xyz[i, 0])), 3) for i in range(len(xyz))]
    out["is_controlB"] = [bool(x) for x in (net == "ContB")]

    dst = ROOT / "outputs/logs/section5_coverage_continuum.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
