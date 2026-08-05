"""§5 coverage collapse, re-tested with the repo's spatial-autocorrelation (spin) null.

The manuscript's Fig 5b reports the sensorimotor->association coverage gap under a
*permuted-axis* null (shuffle the myelin labels), which does NOT preserve spatial
autocorrelation and therefore over-states significance (p = 3.4e-7). Here we re-test
the SAME statistic with `otter.eval.nulls.spin_null` (Alexander-Bloch / Vazquez-
Rodriguez), which rotates parcel centroids on a sphere and so keeps spatial
smoothness in the null. Two statistics are reported:

  (1) continuous Pearson r between per-parcel coverage and the myelin axis
  (2) the sensorimotor-tertile minus association-tertile coverage gap (Fig 5b)

both with a proper spin p.

RESULT ON THE CANONICAL COUPLING (pi_canonical.npy, 2026-07-18): BOTH statistics are
null. Continuous r = +0.14 (spin p = 0.17); tertile gap = 0.68 log units (spin p = 0.29).
On the RETIRED pre-warp coupling (pi_fc_plus_SC_with_all_packs.npy) the tertile gap was
6.74 log units at spin p = 0.002, and that is what the earlier "spin-ROBUST tertile
contrast" note in this file described. It does not survive the canonical coupling. Do
not reinstate the tertile contrast as a positive result without re-deriving it here.

Run: cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/01_coverage_nulls.py
Writes outputs/logs/section5_coverage_nulls.json
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance
from otter.eval.nulls import spin_null, _haar_rotation

DATA = ROOT / "data_external"
N_SPIN = 1000
SEED = 0


def tertile_gap_spin(cov, axis, xyz, n_trials=N_SPIN, seed=SEED):
    """Spin p for the (high-axis tertile - low-axis tertile) coverage gap, using
    the same Haar-rotation scheme as otter.eval.nulls.spin_null. Tertiles are fixed
    by `axis`; only `cov` is spun, so the null preserves spatial autocorrelation."""
    from scipy.spatial import cKDTree
    o = np.argsort(axis); t = len(o) // 3
    lo, hi = o[:t], o[-t:]                       # lo axis (association) vs hi axis (sensorimotor)
    obs = cov[hi].mean() - cov[lo].mean()
    c = xyz - np.nanmean(xyz, 0)
    sph = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(seed); null = np.empty(n_trials)
    for i in range(n_trials):
        _, perm = cKDTree(sph @ _haar_rotation(rng).T).query(sph)
        cp = cov[perm]; null[i] = cp[hi].mean() - cp[lo].mean()
    an = np.abs(null)
    return {"gap_observed": float(obs),
            "p_spin": float((np.sum(an >= abs(obs)) + 1) / (n_trials + 1)),
            "null_abs_mean": float(an.mean()), "null_abs_p95": float(np.percentile(an, 95))}


def main():
    pi = load_pi()
    prov = pi_provenance()
    print(f"π: {prov['pi_file']}  sha256={prov['pi_sha256']}")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    node_region = np.asarray(json.loads((DATA / "human_sc_meta.json").read_text())["node_region"], int)

    coverage = np.log10(np.maximum(pi.sum(0), 1e-300))

    # sensorimotor->association axis: HCP T1w/T2w myelin per Schaefer region
    myelin_reg = {}
    with open(DATA / "fulcher_2019_gradients/human_myelinmap_schaefer400_OTTERorder.csv") as f:
        for row in csv.DictReader(f):
            myelin_reg[int(row["otter_region_id"])] = float(row["t1t2_myelin"])
    myelin = np.array([myelin_reg.get(r, np.nan) for r in node_region])

    ctx = np.isfinite(myelin)
    cov, mye, xyzc = coverage[ctx], myelin[ctx], xyz[ctx]

    cont = spin_null(cov, mye, xyzc, n_trials=N_SPIN, seed=SEED)          # continuous r
    tert = tertile_gap_spin(cov, mye, xyzc)                               # Fig 5b statistic

    print(f"n cortical parcels = {ctx.sum()}")
    print(f"[continuous]  coverage vs myelin: r = {cont['r_observed']:+.3f}   "
          f"spin p = {cont['p_spin']:.4f}  (null|r|95 = {cont['null_abs_p95']:.3f})")
    print(f"[tertile]     sensorimotor - association gap = {tert['gap_observed']:.2f} log units   "
          f"spin p = {tert['p_spin']:.4f}  (null|gap|95 = {tert['null_abs_p95']:.2f})")

    out = {**prov, "n_cortical_parcels": int(ctx.sum()),
           "n_spin": N_SPIN, "coverage_vs_myelin_continuous": cont,
           "coverage_collapse_tertile": tert,
           "note": ("On the canonical coupling BOTH statistics are null under the spin: "
                    "continuous r spin p ~ 0.17, tertile gap spin p ~ 0.29. The 6.7-log-unit "
                    "tertile gap (spin p ~ 0.002) was a property of the retired pre-warp "
                    "coupling pi_fc_plus_SC_with_all_packs.npy, not of the canonical one.")}
    (ROOT / "outputs/logs/section5_coverage_nulls.json").write_text(json.dumps(out, indent=2))
    print("wrote outputs/logs/section5_coverage_nulls.json")


if __name__ == "__main__":
    main()
