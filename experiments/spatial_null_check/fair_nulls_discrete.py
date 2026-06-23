"""Re-examine the DISCRETE results (network bridge) and the gradient under
spatially-fair nulls (audit, after the Margulies spin finding).

Two checks:
  1. Margulies gradient via the fair TRANSLATION null (spin the mouse input,
     route through the real π), confirms the gradient is n.s. (~p=0.22).
  2. Test 1 network bridge (4/8 diagonal-argmax) via a mouse-parcel SPIN null:
     rotate the mouse parcels (so mouse networks keep their spatial shape but
     move location), re-aggregate π, recount diagonal-argmax. Tests whether the
     SPECIFIC mouse-network→human-network correspondence beats spatially-rotated
     network assignments, the discrete analogue of a spin test.

Usage:
    PYTHONPATH=src python experiments/spatial_null_check/fair_nulls_discrete.py
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "autism_subtypes"))

from homer.data import load_cached                                  # noqa: E402
from homer.eval.nulls import translation_spin_null, _haar_rotation  # noqa: E402

ncv = import_module("01_network_crossvalidation")

TARGET_PAIRS = [
    ("Visual", "Visual"), ("Auditory", "Auditory"), ("SomatoMotor", "SomatoMotor"),
    ("DMN", "DMN"), ("Salience", "Salience"), ("HC_Limbic", "Limbic"),
    ("Subcortical", "Subcortical"), ("BF_Olfactory", "Subcortical"),
]


def _sphere(c):
    c = c - np.nanmean(c, axis=0)
    n = np.linalg.norm(c, axis=1, keepdims=True); n[n == 0] = 1.0
    return c / n


def _diag_count(pi, mouse_net, mouse_names, human_net, human_names):
    N = ncv.compute_network_mapping(pi, mouse_net, human_net,
                                    n_mouse=len(mouse_names), n_human=len(human_names))
    sc = ncv.score_mapping(N, mouse_names, human_names, target_pairs=TARGET_PAIRS)
    return sum(1 for r in sc["per_pair"] if r.get("is_argmax_diagonal"))


def main():
    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = np.load(ROOT / "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy").astype(np.float64)

    # ---- 1. Margulies gradient, fair translation null ----
    d = json.loads((ROOT / "outputs/logs/margulies_2016_gradient.json").read_text())
    mouse_g = np.array(d["mouse_gradient"], float)
    human_g = np.array(d["human_gradient"], float)
    mc = M.var[["x", "y", "z"]].to_numpy(float)
    res = translation_spin_null(mouse_g, human_g, pi, mc, n_trials=1000, seed=0)
    print("1. Margulies gradient, fair TRANSLATION null (spin mouse input, real π):")
    print(f"   observed |r| = {abs(res['r_observed']):.3f}   null |r| mean {res['null_abs_mean']:.3f} "
          f"(95th {res['null_abs_p95']:.3f})   p = {res['p_translation_spin']:.3f}")

    # ---- 2. Network bridge, mouse-parcel spin null ----
    mnet, mnames = ncv.assign_mouse_paper_networks(M.var, separate_aud=True)
    hnet, hnames = ncv.assign_human_paper_networks(H.var, separate_aud=True)
    obs = _diag_count(pi, mnet, mnames, hnet, hnames)
    sph = _sphere(mc)
    rng = np.random.default_rng(0)
    n_trials = 500
    null = np.empty(n_trials, dtype=int)
    for t in range(n_trials):
        rot = sph @ _haar_rotation(rng).T
        _, perm = cKDTree(rot).query(sph)
        null[t] = _diag_count(pi, mnet[perm], mnames, hnet, hnames)
    p = (np.sum(null >= obs) + 1) / (n_trials + 1)
    print("\n2. Network bridge, mouse-parcel SPIN null (rotate mouse networks):")
    print(f"   observed diagonal-argmax = {obs}/8")
    print(f"   spin null: mean {null.mean():.2f}/8, 95th pct {np.percentile(null,95):.0f}/8, max {null.max()}/8")
    print(f"   p (observed >= spin null) = {p:.3f}  → "
          f"{'SURVIVES (specific, not spatial)' if p < 0.05 else 'does NOT clearly survive'}")

    (ROOT / "outputs/logs/fair_nulls_discrete.json").write_text(json.dumps({
        "margulies_translation_spin_p": res["p_translation_spin"],
        "network_bridge_observed_diag": int(obs),
        "network_bridge_spin_null_mean": float(null.mean()),
        "network_bridge_spin_p": float(p),
    }, indent=2))


if __name__ == "__main__":
    main()
