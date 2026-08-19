#!/usr/bin/env python3
"""Warm the entropic temperature and ask whether a SMOOTHER coverage field tracks expansion.

epsilon was chosen (docs/02_methods) to maximise anchor-CV accuracy: small eps -> near-hard
matching. That makes coverage a winner-take-all, noisy quantity (L/R reliability 0.22). A warmer
eps softens the coupling. An eps-family is refitted (identical config, only eps varies) and, at
each eps, the following are recorded:
  - sharpness (median top-1 row probability),
  - coverage L/R reliability (Schaefer k vs k+200), the criterion for picking eps,
  - Spearman(region coverage, Xu2020 macaque->human expansion) + spin p,
  - medial-lateral rho.

eps is picked by maximum reliability and the expansion correlation is read there. eps is fixed
by reliability rather than by the expansion p-value.

Writes: outputs/logs/section5_epsilon_sweep.json ; couplings cached in /var/tmp.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached                       # noqa: E402
from otter.models import MultimodalFGW                   # noqa: E402
from otter.eval.nulls import _haar_rotation              # noqa: E402
np.seterr(divide="ignore", invalid="ignore")

EPS = [0.005, 0.02, 0.05, 0.1, 0.2]
N_SPIN = 1000
CACHE = Path("/var/tmp")


def fit_eps(eps, M, H, costs):
    f = CACHE / f"pi_eps_{eps:.3f}.npy"
    if f.exists():
        return np.load(f)
    m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7, epsilon=eps,
                      xyz_weight=0.5, lam_anchor=1.0, alpha=0.5)
    m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"], region_anchors=[])
    pi = m.pi.astype(np.float64)
    np.save(f, pi)
    return pi


def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    b = json.loads((ROOT / "outputs/logs/section5_evolution_battery.json").read_text())
    xu = dict(zip(np.asarray(b["Xu2020 macaque→human expansion"]["schaefer_ids"], int),
                  np.asarray(b["Xu2020 macaque→human expansion"]["map_values"], float)))

    ids = [k for k in range(1, 401) if (nr == k).any()]
    cen = {k: xyz[nr == k].mean(0) for k in ids}
    ev_ids = [k for k in ids if k in xu]
    ev = np.array([xu[k] for k in ev_ids])
    Cev = np.array([cen[k] for k in ev_ids])
    # spin perms on the Xu id set
    c = Cev - Cev.mean(0); sph = c / np.linalg.norm(c, axis=1, keepdims=True)
    tree = cKDTree(sph); rng = np.random.default_rng(0)
    perms = [tree.query(sph @ _haar_rotation(rng).T)[1] for _ in range(N_SPIN)]

    def region_cov(pi):
        col = pi.sum(0)
        return {k: np.log10(col[nr == k].mean() + 1e-300) for k in ids}

    out = {"_criterion": "pick eps by MAX coverage L/R reliability, then read expansion corr there.",
           "eps_values": EPS, "per_eps": {}}

    for eps in EPS:
        t = time.time()
        pi = fit_eps(eps, M, H, costs)
        rc = region_cov(pi)
        # sharpness
        rows = pi / pi.sum(1, keepdims=True)
        sharp = float(np.median(rows.max(1)))
        # L/R reliability: Schaefer k (1..200 L) vs k+200 (R)
        pairs = [(rc[k], rc[k + 200]) for k in range(1, 201) if k in rc and (k + 200) in rc]
        a, bb = np.array(pairs).T
        rel = float(spearmanr(a, bb).statistic)
        # expansion corr + spin
        cev = np.array([rc[k] for k in ev_ids])
        rho = float(spearmanr(cev, ev).statistic)
        null = np.array([spearmanr(cev[p], ev).statistic for p in perms])
        pspin = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (N_SPIN + 1))
        # medial-lateral
        absx = np.abs(np.array([cen[k][0] for k in ev_ids]))
        rho_x = float(spearmanr(cev, absx).statistic)
        out["per_eps"][f"{eps:.3f}"] = {"sharpness_median_top1": sharp, "coverage_LR_reliability": rel,
                                        "expansion_spearman": rho, "expansion_spin_p": pspin,
                                        "medial_lateral_rho": rho_x}
        print(f"eps={eps:.3f}  sharp={sharp:.2f}  reliab={rel:+.2f}  "
              f"exp_rho={rho:+.3f} (p={pspin:.3f})  ML_rho={rho_x:+.3f}   [{time.time()-t:.0f}s]")

    # pick eps by reliability
    best = max(out["per_eps"].items(), key=lambda kv: kv[1]["coverage_LR_reliability"])
    out["chosen_eps_by_reliability"] = best[0]
    out["expansion_at_chosen_eps"] = {"spearman": best[1]["expansion_spearman"],
                                      "spin_p": best[1]["expansion_spin_p"],
                                      "reliability": best[1]["coverage_LR_reliability"]}
    print(f"\nchosen eps (max reliability) = {best[0]}: expansion rho={best[1]['expansion_spearman']:+.3f} "
          f"spin p={best[1]['expansion_spin_p']:.3f}")

    dst = ROOT / "outputs/logs/section5_epsilon_sweep.json"
    dst.write_text(json.dumps(out, indent=2)); print(f"wrote {dst}")


if __name__ == "__main__":
    main()
