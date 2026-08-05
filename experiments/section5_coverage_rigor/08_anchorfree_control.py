#!/usr/bin/env python3
"""Section 5, rebuilt: coverage on a scale-free footing, with an ANCHOR-FREE control.

WHY THIS SCRIPT EXISTS
----------------------
The original section-5 headline ("coverage collapses 6.7 log-units from sensorimotor to
association cortex; 53 % of parcels receive negligible mass") does not survive scrutiny:

1.  Coverage was defined per parcel as log10(max(pi.sum(0), 1e-300)) and then AVERAGED IN
    LOG SPACE. Entropic OT gives pi_ij ~ exp(-C_ij / eps), so with eps = 5e-3 the log of the
    mass is the transport COST rescaled by 1/eps. The observed 91-log-unit spread implies a
    cost spread of 1.05, and the costs are max-normalised to exactly [0, 1]. The quantity
    being reported as "mouse mass received" was a cost readout, exponentially amplified.

2.  No human parcel is actually uncovered. The minimum column mass is 7.2e-94. "53 %
    uncovered" is a statement about where a threshold is placed, not about biology, and the
    figure moves to 41 % / 58 % under other equally arbitrary thresholds.

3.  The 6.7 log-unit gap therefore lives in the numerical underflow tail. Floored at 1e-6
    (the paper's own definition of negligible) it is 0.27 log-units, and it no longer clears
    a spin null. In linear mass, sensorimotor cortex receives only 1.5x association cortex.

WHAT SURVIVES, AND THE CONTROL THAT ESTABLISHES IT
--------------------------------------------------
Standardised (SD) tertile gaps are scale-free and DO hold. The obvious objection to them is
that coverage might simply track distance from the curated anchors: rho(coverage, anchor
distance) = -0.31, three times its correlation with the cortical hierarchy (+0.09).

So we re-fit the model with EVERY anchor removed (garin_anchor cleared, zero region packs)
and recompute. If the association deficit were an artefact of where we placed supervision,
stripping the supervision would destroy it.

It does the opposite. The deficit GROWS (+0.47 -> +0.54 SD), as does the dissociation
(+0.64 -> +0.71 SD). The p-values rise only because the anchor-free map is spatially noisier,
which widens the spin null (95th pct 0.36 -> 0.55); the effect itself is larger. The curated
packs, several of which deliberately supply coverage to association territory (lateral PFC,
PPC, cingulate), slightly ATTENUATE the deficit rather than creating it.

NOTE ON THE ABLATION COUPLINGS ALREADY ON DISK
----------------------------------------------
Do NOT use outputs/coupling/pi_ablation_xyz_only.npy as the anchor-free model. It was fitted
with lam_anchor=0, and in supervised.py that does `M[mp, :] = lam` -- with lam = 0 it zeroes
the ENTIRE cross-species cost row for the 42 anchor parcels rather than removing the anchors,
leaving them free to go anywhere. Its sidecar JSON still reports n_visible_anchors = 21. The
correct way to drop anchors is to clear M.var["garin_anchor"], which is what we do below (and
what ablation_ladder.py does).

Run:  cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/08_anchorfree_control.py
Writes: outputs/coupling/pi_anchorfree_control.npy
        outputs/logs/section5_anchorfree_control.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached, load_pi, pi_provenance  # noqa: E402
from otter.data.anchors import get_anchor_index                 # noqa: E402
from otter.eval.nulls import _haar_rotation                     # noqa: E402

N_SPIN = 1000
SEED = 0
PI_AF = ROOT / "outputs/coupling/pi_anchorfree_control.npy"


def fit_anchor_free():
    """GW(FC+SC) + xyz, with garin_anchor CLEARED and no region packs."""
    from otter.models import MultimodalFGW

    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")

    g0m = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    g0h = H.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    M.var["garin_anchor"] = g0m & False          # remove the anchors properly
    H.var["garin_anchor"] = g0h & False

    m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7, epsilon=5e-3,
                      xyz_weight=0.5, lam_anchor=1.0, alpha=0.5)
    m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"], region_anchors=[])

    M.var["garin_anchor"] = g0m
    H.var["garin_anchor"] = g0h
    PI_AF.parent.mkdir(parents=True, exist_ok=True)
    np.save(PI_AF, m.pi.astype(np.float64))
    print(f"fitted and saved anchor-free coupling -> {PI_AF}")
    return m.pi.astype(np.float64)


def molecular_similarity(n_human: int) -> np.ndarray:
    """Best transcriptomic match to any mouse parcel, per human parcel (51-gene panel)."""
    D = ROOT / "data_external"
    mg, hg = np.load(D / "mouse_genes_aligned.npy"), np.load(D / "human_genes_aligned.npy")

    def zc(A):
        A = A.astype(float)
        mu, sd = np.nanmean(A, 0), np.nanstd(A, 0)
        sd[sd < 1e-9] = 1.0
        return (A - mu) / sd

    Mz, Hz = zc(mg), zc(hg)
    mok, hok = np.isfinite(Mz).all(1), np.isfinite(Hz).all(1)
    Mc = Mz[mok] - Mz[mok].mean(1, keepdims=True)
    Mn = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
    Hc = Hz - np.nanmean(Hz, 1, keepdims=True)
    Hn = np.full_like(Hc, np.nan)
    Hn[hok] = Hc[hok] / np.linalg.norm(Hc[hok], axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        return np.nanmax(Hn @ Mn.T, axis=1)


def spin_perms(coords, n_trials=N_SPIN, seed=SEED):
    c = coords - np.nanmean(coords, 0)
    sph = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
    tree = cKDTree(sph)
    rng = np.random.default_rng(seed)
    return [tree.query(sph @ _haar_rotation(rng).T)[1] for _ in range(n_trials)]


def tertile_gap(sig, hi, lo):
    return float(sig[hi].mean() - sig[lo].mean())


def gap_with_spin(sig, hi, lo, perms):
    obs = tertile_gap(sig, hi, lo)
    null = np.array([tertile_gap(sig[p], hi, lo) for p in perms])
    an = np.abs(null)
    return {"gap_sd": obs,
            "p_spin": float((np.sum(an >= abs(obs)) + 1) / (len(perms) + 1)),
            "null_abs_p95": float(np.percentile(an, 95))}


def z(v):
    return (v - v.mean()) / v.std()


def main():
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    bk = json.loads((ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())
    mye = np.asarray(bk["myelin_per_parcel"], float)

    pi_prod = load_pi()
    pi_af = np.load(PI_AF) if PI_AF.exists() else fit_anchor_free()
    best = molecular_similarity(len(xyz))
    adist = cKDTree(xyz[get_anchor_index(H.var).pos]).query(xyz)[0]

    out = {"_why": ("Coverage is reported as a SCALE-FREE standardised gap, never as log-units. "
                    "log10(pi column mass) is a transport-cost readout (pi ~ exp(-C/eps), "
                    "eps=5e-3), not a mass; the old 6.7 log-unit gap lived in the underflow "
                    "tail and does not survive a physical floor or a rank transform."),
           "epsilon": 5e-3,
           "min_column_mass": float(pi_prod.sum(0).min()),
           "n_parcels_truly_zero": int((pi_prod.sum(0) == 0).sum())}

    # ---- support concentration: threshold-free ----------------------------------
    p = pi_prod.sum(0) / pi_prod.sum()
    s = np.sort(p)[::-1]
    cum = np.cumsum(s)
    out["support"] = {
        "parcels_holding_90pct_of_mass": int(np.searchsorted(cum, 0.90) + 1),
        "parcels_holding_95pct_of_mass": int(np.searchsorted(cum, 0.95) + 1),
        "effective_support_1_over_sum_p2": float(1.0 / np.sum(p ** 2)),
        "n_human_parcels": int(len(p)),
    }

    # ---- the tertile / dissociation battery, both couplings ----------------------
    cortex = np.isfinite(mye)
    out["confound"] = {
        "rho_coverage_vs_anchor_distance_production": float(
            spearmanr(np.log10(np.maximum(pi_prod.sum(0), 1e-300))[cortex], adist[cortex]).statistic),
        "rho_coverage_vs_anchor_distance_anchorfree": float(
            spearmanr(np.log10(np.maximum(pi_af.sum(0), 1e-300))[cortex], adist[cortex]).statistic),
        "_note": ("The anchor-free model contains NO anchors, so its residual correlation with "
                  "anchor distance is biology: we placed anchors where the connectomes "
                  "independently agree."),
    }

    for label, pi in (("production", pi_prod), ("anchor_free", pi_af)):
        cov = np.log10(np.maximum(pi.sum(0), 1e-300))
        m = np.isfinite(cov) & np.isfinite(best) & np.isfinite(mye)
        perms = spin_perms(xyz[m])
        o = np.argsort(mye[m])
        t = len(o) // 3
        lo, hi = o[:t], o[-t:]
        cz, bz = z(cov[m]), z(best[m])
        out[label] = {
            "n_parcels": int(m.sum()),
            "connectivity_coverage_gap": gap_with_spin(cz, hi, lo, perms),
            "transcriptomic_similarity_gap": gap_with_spin(bz, hi, lo, perms),
            "dissociation_gap": gap_with_spin(cz - bz, hi, lo, perms),
            "rho_coverage_vs_hierarchy": float(spearmanr(cov[m], mye[m]).statistic),
        }
        print(f"{label:12s} coverage {out[label]['connectivity_coverage_gap']['gap_sd']:+.2f} SD "
              f"(p={out[label]['connectivity_coverage_gap']['p_spin']:.3f})   "
              f"dissociation {out[label]['dissociation_gap']['gap_sd']:+.2f} SD "
              f"(p={out[label]['dissociation_gap']['p_spin']:.3f})")

    # ---- scale sensitivity: what the old headline was actually measuring ---------
    covm, myem = pi_prod.sum(0)[cortex], mye[cortex]
    q = np.quantile(myem, [1 / 3, 2 / 3])
    sens, asso = myem >= q[1], myem <= q[0]
    out["scale_sensitivity_of_the_retired_claim"] = {
        "gap_log_units_floor_1e-300_AS_PUBLISHED": float(
            np.log10(np.maximum(covm, 1e-300))[sens].mean() - np.log10(np.maximum(covm, 1e-300))[asso].mean()),
        "gap_log_units_floor_1e-12": float(
            np.log10(np.maximum(covm, 1e-12))[sens].mean() - np.log10(np.maximum(covm, 1e-12))[asso].mean()),
        "gap_log_units_floor_1e-6": float(
            np.log10(np.maximum(covm, 1e-6))[sens].mean() - np.log10(np.maximum(covm, 1e-6))[asso].mean()),
        "linear_mass_ratio_sensorimotor_over_association": float(covm[sens].mean() / covm[asso].mean()),
        "spearman_coverage_vs_hierarchy_scale_invariant": float(spearmanr(covm, myem).statistic),
    }

    # coverage percentile map for the figure (scale-free, what panel a now plots)
    pct = np.full(len(xyz), np.nan)
    pct[cortex] = rankdata(pi_prod.sum(0)[cortex]) / cortex.sum() * 100.0
    out["coverage_percentile_cortex"] = [None if not np.isfinite(v) else round(float(v), 3) for v in pct]

    dst = ROOT / "outputs/logs/section5_anchorfree_control.json"
    out.update(pi_provenance())   # which coupling produced these numbers
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
