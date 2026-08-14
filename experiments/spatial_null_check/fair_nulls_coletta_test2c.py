"""Mouse-parcel SPIN null for the remaining discrete/network validations:
Coletta 2020 cross-species RSN correspondence and Pagani Test 2c (07_full_matrix).

Rotate the mouse parcels on a sphere (mouse networks keep their spatial shape but
move location), re-aggregate π, recompute the statistic. Tests whether the
SPECIFIC mouse→human correspondence beats spatially-rotated assignments.

    PYTHONPATH=src python experiments/spatial_null_check/fair_nulls_coletta_test2c.py
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "autism_subtypes"))
sys.path.insert(0, str(ROOT / "experiments" / "coletta_2020_cross_species_rsn"))

from otter.data import load_cached, load_pi, pi_provenance                          # noqa: E402
from otter.data.anchors import get_anchor_index             # noqa: E402
from otter.data.networks import assign_networks, NETWORKS   # noqa: E402
from otter.eval.nulls import _haar_rotation                 # noqa: E402

cm = import_module("01_correspondence_validation")
nc = import_module("01_network_crossvalidation")
fm = import_module("07_full_matrix_translation")
st = import_module("04_subtype_translation")


def _sphere(c):
    c = c - np.nanmean(c, axis=0)
    n = np.linalg.norm(c, axis=1, keepdims=True); n[n == 0] = 1.0
    return c / n


def _spin_perms(coords, n, seed=0):
    sph = _sphere(coords); rng = np.random.default_rng(seed)
    for _ in range(n):
        rot = sph @ _haar_rotation(rng).T
        yield cKDTree(rot).query(sph)[1]


def main():
    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = load_pi().astype(np.float64)   # canonical coupling. Repointed 2026-07-20: this was
    # hardcoded to the retired pi, so the spin null was computed on a
    # different coupling than the observed statistic it was compared against.
    mc = M.var[["x", "y", "z"]].to_numpy(float)
    out = {}

    # ---------- Coletta cross-species RSN (diagonal-argmax) ----------
    idx_m = get_anchor_index(M.var)
    mouse_net = assign_networks(M.var, idx_m)
    human_net, hnames = nc.assign_human_paper_networks(H.var, separate_aud=True)
    a = hnames.index("Auditory"); s = hnames.index("SomatoMotor")
    human_net = human_net.copy(); human_net[human_net == a] = s
    target_pairs = [("sensorimotor", "SomatoMotor"), ("visual", "Visual"),
                    ("auditory", "Auditory"), ("salience", "Salience"),
                    ("frontal_dmn", "DMN"), ("temporal_dmn", "DMN"),
                    ("limbic", "Limbic"), ("frontoparietal", "DorsAtten"),
                    ("subcortical", "Subcortical"), ("olfactory", "Limbic")]

    def coletta_diag(mnet):
        _, _, score = cm.labeled_correspondence(pi, mnet, NETWORKS, human_net,
                                                 hnames, target_pairs)
        recs = score["per_pair"] if isinstance(score, dict) else score
        return sum(1 for r in recs if r.get("is_argmax_diagonal"))

    obs_c = coletta_diag(mouse_net)
    null_c = np.array([coletta_diag(mouse_net[p]) for p in _spin_perms(mc, 500)])
    p_c = (np.sum(null_c >= obs_c) + 1) / 501
    print(f"Coletta RSN: observed {obs_c}/10 diagonal-argmax | spin null mean "
          f"{null_c.mean():.2f}/10 (95th {np.percentile(null_c,95):.0f}) | p={p_c:.3f}  "
          f"{'SURVIVES' if p_c < 0.05 else 'n.s.'}")
    out["coletta"] = {"observed": int(obs_c), "spin_null_mean": float(null_c.mean()),
                       "spin_p": float(p_c)}

    # ---------- Pagani Test 2c (07_full_matrix, 36-element Pearson) ----------
    m_net, m_names = fm.assign_mouse_pagani_networks(M.var)
    hnet2, hpaper = nc.assign_human_paper_networks(H.var, separate_aud=True)
    a2 = hpaper.index("Auditory"); s2 = hpaper.index("SomatoMotor")
    hnet2 = hnet2.copy(); hnet2[hnet2 == a2] = s2
    pag_h = ["Control", "DMN", "DorsAtten", "Limbic", "Salience", "SomatoMotor",
             "Visual", "Subcortical"]
    h2idx = {n: hpaper.index(n) for n in pag_h if n in hpaper}
    new_h = np.full_like(hnet2, -1)
    for ni, n in enumerate(pag_h):
        if n in h2idx:
            new_h[hnet2 == h2idx[n]] = ni
    data = st.load_pagani_subtype_matrices()
    d_mouse = fm.symmetrise(data["mouse_hyper"]) - fm.symmetrise(data["mouse_hypo"])
    d_humb = fm.symmetrise(data["human_hyper"]) - fm.symmetrise(data["human_hypo"])
    obs_flat = fm.upper_triangle_flat(d_humb)

    def t2c_r(mnet):
        keep = mnet >= 0
        T = fm.build_translation_operator(pi[keep], mnet[keep], len(m_names),
                                          new_h, len(pag_h))
        pred = fm.upper_triangle_flat(T.T @ d_mouse @ T)
        return pearsonr(pred, obs_flat)[0]

    obs_r = t2c_r(m_net)
    null_r = np.array([abs(t2c_r(m_net[p])) for p in _spin_perms(mc, 500)])
    p_t = (np.sum(null_r >= abs(obs_r)) + 1) / 501
    print(f"Test 2c   : observed Pearson r={obs_r:+.3f} | spin null |r| mean "
          f"{null_r.mean():.3f} (95th {np.percentile(null_r,95):.3f}) | p={p_t:.3f}  "
          f"{'SURVIVES' if p_t < 0.05 else 'n.s.'}")
    out["test_2c"] = {"observed_r": float(obs_r), "spin_null_abs_mean": float(null_r.mean()),
                       "spin_p": float(p_t)}

    out.update(pi_provenance())   # which coupling produced these nulls
    (ROOT / "outputs/logs/fair_nulls_coletta_test2c.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
