"""Apply a spatial-autocorrelation-preserving (spin) null to the parcel-level
gradient validations (audit finding F-024).

The experiments report significance against a *permuted-π* null, which destroys
spatial autocorrelation and so over-states significance when the target is a
smooth map (a gradient). This script recomputes the significance with the spin
null in `homer.eval.nulls.spin_null` (rotate parcel centroids on a sphere) and
prints both p-values side by side.

Usage:
    PYTHONPATH=src python experiments/spatial_null_check/apply_spin_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from homer.data import load_cached          # noqa: E402
from homer.eval.nulls import spin_null      # noqa: E402

LOG = ROOT / "outputs" / "logs"


def _self_test(coords):
    """Sanity check the spin null: a smooth map vs itself must be spin-significant;
    a smooth map vs an independent smooth map must NOT be."""
    rng = np.random.default_rng(0)
    # build two independent smooth maps by averaging each parcel's value over its
    # spatial neighbours (induces autocorrelation)
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    nbr = tree.query(coords, k=40)[1]
    def smooth():
        v = rng.standard_normal(len(coords))
        return v[nbr].mean(1)
    m1 = smooth(); m2 = smooth()
    s_self = spin_null(m1, m1, coords, n_trials=500, seed=1)
    s_indep = spin_null(m1, m2, coords, n_trials=500, seed=2)
    print("  self-test (smooth vs itself): r=%.2f  p_spin=%.4f  (expect p≈0)"
          % (s_self["r_observed"], s_self["p_spin"]))
    print("  self-test (smooth vs indep ): r=%.2f  p_spin=%.3f  (expect p NOT small)"
          % (s_indep["r_observed"], s_indep["p_spin"]))


def main():
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    coords = H.var[["x", "y", "z"]].to_numpy(dtype=float)
    print(f"human parcels: {len(coords)}")

    print("\nSpin-null self-test:")
    _self_test(coords)

    out = {}
    for name, fname, akey, bkey in [
        ("Margulies gradient", "margulies_2016_gradient.json",
         "human_gradient", "predicted_human_gradient"),
    ]:
        p = LOG / fname
        if not p.exists():
            print(f"\n{name}: log missing ({fname}) — skip"); continue
        d = json.loads(p.read_text())
        a = np.array(d[akey], dtype=float)
        b = np.array(d[bkey], dtype=float)
        if a.shape[0] != len(coords):
            print(f"\n{name}: vector len {a.shape[0]} != {len(coords)} parcels — skip"); continue
        res = spin_null(a, b, coords, n_trials=1000, seed=0)
        # permuted-π p from the log for comparison
        perm_p = None
        nd = d.get("null", {})
        for k in ("empirical_p", "p", "abs_empirical_p"):
            if isinstance(nd, dict) and k in nd:
                perm_p = nd[k]; break
        n_finite = int((np.isfinite(a) & np.isfinite(b)).sum())
        print(f"\n{name}  (n={n_finite} parcels)")
        print(f"  observed |Pearson r|     = {abs(res['r_observed']):.3f}")
        print(f"  permuted-π null |r| p    = {perm_p if perm_p is not None else 'n/a (log)'}")
        print(f"  SPIN null p (two-sided)  = {res['p_spin']:.4f}")
        print(f"  spin null |r|: mean {res['null_abs_mean']:.3f}, 95th pct {res['null_abs_p95']:.3f}")
        verdict = ("survives" if res["p_spin"] < 0.05 else "does NOT survive")
        print(f"  → under a spatial null, the gradient correlation {verdict} (p={res['p_spin']:.3f}).")
        out[name] = {"abs_r": abs(res["r_observed"]), "p_spin": res["p_spin"],
                      "spin_null_abs_mean": res["null_abs_mean"], "n": n_finite}

    out_path = LOG / "spin_test_gradients.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
