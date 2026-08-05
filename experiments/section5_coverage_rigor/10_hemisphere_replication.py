#!/usr/bin/env python3
"""Hemispheric replication of the coverage deficit, and the retirement of the Hill result.

WHY
---
The Hill 2010 macaque->human expansion map is RIGHT-HEMISPHERE ONLY (Schaefer ids 201-400).
It was published as "coverage aligns with macaque->human expansion, rho = -0.18, spin p = 0.046".
Two things are wrong with that.

1.  The spin null was computed by projecting a single hemisphere of centroids onto a sphere and
    applying a Haar rotation. That is not a valid spin: the point cloud occupies a half-sphere,
    so rotations fold it back onto itself and the null inflates (its 95th percentile is 0.21,
    against 0.13-0.14 for the whole-brain maps).

2.  OTTER's human atlas is left/right symmetric, so the map can be mirrored and re-tested.
    The correlation does not replicate. It reverses:

        RIGHT hemisphere (Hill's native side) : rho = -0.184  (n = 197)
        LEFT  hemisphere (Hill mirrored)      : rho = +0.055  (n = 191)

    A real signature of cortical expansion would not be one-sided. Hill is retired.

That test is only decisive if the CLAIM WE KEEP survives it. So we apply the same check to the
coverage deficit and the connectional-vs-molecular dissociation. They do survive: the effect
size is essentially unchanged in each hemisphere taken alone.

We deliberately do NOT report spin p values for the single-hemisphere fits, for the same reason
we discarded Hill's: a Haar rotation of a half-sphere is not a spin test. The effect SIZE is the
evidence here.

Incidental finding worth knowing: coverage on the left correlates with coverage on the mirrored
right at only rho = +0.22. Semi-relaxed OT never constrains the human marginal, so nothing forces
the coupling to be bilaterally symmetric, and entropic amplification lets small cost asymmetries
send mass preferentially to one side. Coverage is therefore noisy per parcel, which is a further
reason to report it as a tertile contrast rather than a per-parcel map.

Run:  cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/10_hemisphere_replication.py
Writes: outputs/logs/section5_hemisphere_replication.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached, load_pi, pi_provenance  # noqa: E402
from otter.eval.nulls import _haar_rotation              # noqa: E402

N_SPIN = 1000


def molecular_similarity():
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


def main():
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    mye = np.asarray(json.loads(
        (ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text()
    )["myelin_per_parcel"], float)
    pi = load_pi()
    cov = np.log10(np.maximum(pi.sum(0), 1e-300))
    best = molecular_similarity()

    out = {"_why": ("Hill's macaque->human expansion correlation REVERSES between hemispheres "
                    "(-0.18 right, +0.06 left with the map mirrored) and is retired. The "
                    "coverage deficit and the dissociation do NOT reverse: their effect size "
                    "is stable in each hemisphere alone. Single-hemisphere spin p values are "
                    "deliberately omitted: a Haar rotation of a half-sphere is not a spin."),
           "n_spin_whole_brain": N_SPIN}

    def block(sel, label, with_p):
        m = sel & np.isfinite(cov) & np.isfinite(best) & np.isfinite(mye)
        c, b, ax_, co = cov[m], best[m], mye[m], xyz[m]
        z = lambda v: (v - v.mean()) / v.std()          # noqa: E731
        cz, bz = z(c), z(b)
        o = np.argsort(ax_)
        t = len(o) // 3
        lo, hi = o[:t], o[-t:]
        gap = lambda s: float(s[hi].mean() - s[lo].mean())   # noqa: E731

        res = {"n": int(m.sum()),
               "connectivity_coverage_gap": gap(cz),
               "transcriptomic_similarity_gap": gap(bz),
               "dissociation_gap": gap(cz - bz)}

        if with_p:                                       # whole brain only: a valid sphere
            cc = co - co.mean(0)
            sph = cc / np.linalg.norm(cc, axis=1, keepdims=True)
            tree = cKDTree(sph)
            rng = np.random.default_rng(0)
            perms = [tree.query(sph @ _haar_rotation(rng).T)[1] for _ in range(N_SPIN)]
            for key, sig in (("connectivity_coverage_gap", cz),
                             ("transcriptomic_similarity_gap", bz),
                             ("dissociation_gap", cz - bz)):
                obs = gap(sig)
                null = np.abs([gap(sig[p]) for p in perms])
                res[key + "_spin_p"] = float((np.sum(null >= abs(obs)) + 1) / (N_SPIN + 1))
        else:
            res["_no_spin_p"] = ("A Haar rotation of a single-hemisphere point cloud folds onto "
                                 "a half-sphere and is not a valid spin test. Effect size only.")
        print(f"  {label:<12} n={res['n']:>4}   coverage {res['connectivity_coverage_gap']:+.2f}"
              f"   dissociation {res['dissociation_gap']:+.2f}")
        return res

    print("hemispheric replication of the coverage deficit:")
    out["both"] = block(np.ones(len(cov), bool), "both", True)
    out["left"] = block(xyz[:, 0] < 0, "left", False)
    out["right"] = block(xyz[:, 0] > 0, "right", False)

    # ---- the Hill reversal, recorded so it cannot quietly come back ------------------
    nr = np.asarray(json.loads(
        (ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    ev = json.loads((ROOT / "outputs/logs/section5_evolution_battery.json").read_text())
    hv = ev.get("Hill2010 macaque→human expansion", {})
    if "schaefer_ids" in hv:
        ids = np.asarray(hv["schaefer_ids"], int)
        hill = dict(zip(ids, np.asarray(hv["map_values"], float)))
        col = pi.sum(0)
        rcov = {k: float(np.log10(col[nr == k].mean() + 1e-300))
                for k in range(1, 401) if (nr == k).any()}
        R = [k for k in ids if k in rcov]
        L = [k - 200 for k in ids if (k - 200) in rcov]
        out["hill_hemisphere_reversal"] = {
            "rho_right_native": float(spearmanr([rcov[k] for k in R],
                                                [hill[k] for k in R]).statistic),
            "rho_left_mirrored": float(spearmanr([rcov[k] for k in L],
                                                 [hill[k + 200] for k in L]).statistic),
            "verdict": "RETIRED: reverses sign between hemispheres; not a robust effect.",
        }
        print(f"\n  Hill: right {out['hill_hemisphere_reversal']['rho_right_native']:+.3f}  "
              f"left(mirrored) {out['hill_hemisphere_reversal']['rho_left_mirrored']:+.3f}"
              f"   -> RETIRED")

    # coverage's own bilateral symmetry (weak; see docstring)
    pairs = [(rcov[k - 200], rcov[k]) for k in range(201, 401)
             if k in rcov and (k - 200) in rcov] if "schaefer_ids" in hv else []
    if pairs:
        a, b = np.asarray(pairs).T
        out["coverage_bilateral_symmetry_rho"] = float(spearmanr(a, b).statistic)
        print(f"  coverage L vs mirrored R: rho = {out['coverage_bilateral_symmetry_rho']:+.3f}")

    dst = ROOT / "outputs/logs/section5_hemisphere_replication.json"
    out.update(pi_provenance())   # which coupling produced these numbers
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
