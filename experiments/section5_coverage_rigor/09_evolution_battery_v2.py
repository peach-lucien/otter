#!/usr/bin/env python3
"""Evolution battery, corrected.

BUG THIS FIXES
--------------
07_evolution_battery.py stored `spearman` (a RANK correlation) alongside `spin_p` taken from
`spin_null`, which computes a PEARSON correlation. The downstream plot then coloured its bars by
that p while printing the rho. The two statistics disagree because coverage's Pearson correlation is
inflated by the entropic-OT underflow tail (see 08_anchorfree_control.py), while its Spearman
is not.

Consequence: the HCP T1w/T2w map was reported at "rho = +0.11, spin p = 0.037" and counted
among the maps that clear the null. Its Spearman spin p is 0.10, i.e. NOT significant; the
0.037 belongs to Pearson r = +0.151. The claim "four of the seven clear a conservative spin
null" is not supportable as published.

This script spin-tests the statistic it reports: a Spearman correlation against a
Spearman-based spin null, on rank-transformed coverage (scale-free, so immune to epsilon).

Run:  cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/09_evolution_battery_v2.py
Writes: outputs/logs/section5_evolution_battery_v2.json
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

from otter.data import load_cached, load_pi, pi_provenance   # noqa: E402
from otter.eval.nulls import _haar_rotation          # noqa: E402

N_SPIN = 1000
SEED = 0


def spearman_spin(a, b, coords, n_trials=N_SPIN, seed=SEED):
    """Spearman rho of a vs b, against a spin null that rotates b. Reports what it tests."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b, coords = a[m], b[m], coords[m]
    obs = float(spearmanr(a, b).statistic)

    c = coords - np.nanmean(coords, 0)
    sph = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
    tree = cKDTree(sph)
    rng = np.random.default_rng(seed)
    null = np.empty(n_trials)
    for i in range(n_trials):
        _, perm = tree.query(sph @ _haar_rotation(rng).T)
        null[i] = spearmanr(a, b[perm]).statistic
    an = np.abs(null)
    return {"spearman": obs,
            "spin_p": float((np.sum(an >= abs(obs)) + 1) / (n_trials + 1)),
            "null_abs_p95": float(np.percentile(an, 95)),
            "n": int(m.sum())}


def main():
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    node_region = np.asarray(json.loads(
        (ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)

    old = json.loads((ROOT / "outputs/logs/section5_evolution_battery.json").read_text())

    # Schaefer-region centroids, to match the region-level arrays 07 persists.
    cen = {k: xyz[node_region == k].mean(0) for k in range(1, 401) if (node_region == k).any()}

    maps = {k: v for k, v in old.items()
            if isinstance(v, dict) and v.get("map_values") is not None}
    if not maps:
        raise SystemExit(
            "section5_evolution_battery.json does not store the per-region map values, so the "
            "battery cannot be recomputed. Re-run 07_evolution_battery.py (which now persists "
            "schaefer_ids / coverage_values / map_values), then re-run this script. Do NOT copy "
            "the old p-values across: they are Pearson p-values attached to Spearman rhos."
        )

    out = {"_why": ("Reports a Spearman rho tested against a Spearman spin null. The previous "
                    "battery attached a Pearson spin p to a Spearman rho. The two disagree "
                    "because coverage's Pearson correlation is inflated by a handful of "
                    "deep-underflow regions, which a rank statistic is immune to."),
           "_source": ("derives entirely from outputs/logs/section5_evolution_battery.json; "
                       "the pi provenance below is that log's, re-stated here."),
           "n_spin": N_SPIN,
           **{k: v for k, v in old.get("_meta", {}).items()
              if k in ("pi_file", "pi_sha256")}}
    for k, v in maps.items():
        ids = np.asarray(v["schaefer_ids"], int)
        # coverage is rank-transformed: scale-free, and it is the statistic actually reported
        cov = rankdata(np.asarray(v["coverage_values"], float)).astype(float)
        mp = np.asarray(v["map_values"], float)
        C = np.array([cen[i] for i in ids], float)
        out[k] = spearman_spin(cov, mp, C)
        out[k]["spin_p_as_published_PEARSON"] = v.get("spin_p")
        flip = ("  <-- FLIPS" if (v.get("spin_p", 1) < 0.05) != (out[k]["spin_p"] < 0.05) else "")
        star = "*" if out[k]["spin_p"] < 0.05 else " "
        print(f"  {k:<40} rho = {out[k]['spearman']:+.3f}   "
              f"spin p = {out[k]['spin_p']:.3f}{star}   "
              f"(as published: {v.get('spin_p', float('nan')):.3f}){flip}")

    n_sig = sum(1 for k in maps if out[k]["spin_p"] < 0.05)
    n_old = sum(1 for k in maps if maps[k].get("spin_p", 1) < 0.05)
    out["_summary"] = {"n_maps": len(maps), "n_clearing_spin_null": n_sig,
                       "n_clearing_as_published": n_old}
    print(f"\n{n_sig} of {len(maps)} maps clear the spin null (as published: {n_old})")

    dst = ROOT / "outputs/logs/section5_evolution_battery_v2.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
