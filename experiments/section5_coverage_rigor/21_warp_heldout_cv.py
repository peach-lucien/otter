#!/usr/bin/env python3
"""Clean held-out anchor CV for the warp: does it help anchors held out of BOTH warp + supervision?

Beauchamp overlaps the Garin anchors, so it can't fairly judge an anchor-fitted warp. Here we use the
anchors themselves as a held-out test, removing the confound:

  k-fold on unique homolog pair_ids. For each fold:
    - test anchors = anchors whose pair_id is in the fold; train = the rest.
    - CLEAR garin_anchor for the test anchors (both species) so they are NOT supervised in the fit.
    - fit the coupling two ways, identical except the spatial cost:
        WARP : M_xyz from a TPS warp fitted on TRAIN anchor coords only.
        EUCL : the naive per-species-normalised Euclidean M_xyz.
    - score recovery of the held-out test anchors: mass within 10 mm of the true human homolog, and
      the xyz distance from the routed argmax to the true human homolog.

If WARP beats EUCL on held-out anchors, the warp generalises (learns the global deformation), not
just memorises. Runs a slice of folds (argv: fold_start fold_end) and appends to the log.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached                       # noqa: E402
from otter.data.anchors import get_anchor_index          # noqa: E402
from otter.models import MultimodalFGW                   # noqa: E402

LOG = ROOT / "outputs/logs/section5_warp_heldout_cv.json"
K = 5
NEAR_MM = 10.0


def warped_M(src, dst, mxyz, hxyz):
    warp = RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=1e-3)
    wm = warp(mxyz)
    d = np.sqrt(((wm[:, None, :] - hxyz[None, :, :]) ** 2).sum(-1))
    return (d / max(d.max(), 1e-9)).astype(np.float64)


def eucl_M(mouse_var, human_var):
    from otter.models.supervised import _build_xyz_M
    M = _build_xyz_M(mouse_var, human_var)
    return M.astype(np.float64)


def score_heldout(pi, test, mxyz, hxyz):
    """test: list of (mouse_idx, human_idx). Returns mean mass-within-10mm and mean argmax dist."""
    massn, dists = [], []
    for mi, hi in test:
        col = pi[mi]
        near = np.linalg.norm(hxyz - hxyz[hi], axis=1) <= NEAR_MM
        massn.append(col[near].sum() / max(col.sum(), 1e-12))
        am = int(np.argmax(col))
        dists.append(float(np.linalg.norm(hxyz[am] - hxyz[hi])))
    return float(np.mean(massn)), float(np.mean(dists))


def main(fs, fe):
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    mxyz = M.var[["x", "y", "z"]].to_numpy(float); hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    im = get_anchor_index(M.var); ih = get_anchor_index(H.var)
    # match homologs by (pair_id, hemisphere)
    hlut = {(int(p), str(h)): int(k) for k, p, h in zip(ih.pos, ih.pair_ids, ih.hemispheres)}
    trip = [(int(mp), hlut[(int(pid), str(hemi))], int(pid))
            for mp, pid, hemi in zip(im.pos, im.pair_ids, im.hemispheres)
            if (int(pid), str(hemi)) in hlut]
    uids = sorted({pid for _, _, pid in trip})
    rng = np.random.default_rng(0); order = rng.permutation(uids)
    folds = [set(order[i::K].tolist()) for i in range(K)]

    apid_m = M.var["anchor_pair_id"].to_numpy()
    apid_h = H.var["anchor_pair_id"].to_numpy()
    g0m = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool).copy()
    g0h = H.var["garin_anchor"].fillna(False).to_numpy().astype(bool).copy()

    def fit(Mx):
        m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7, epsilon=5e-3,
                          xyz_weight=0.5, lam_anchor=1.0, alpha=0.5)
        m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"], M_xyz=Mx, region_anchors=[])
        return m.pi.astype(np.float64)

    out = json.loads(LOG.read_text()) if LOG.exists() else {"_config": f"{K}-fold, held out of warp+supervision, near={NEAR_MM}mm"}
    for f in range(fs, fe):
        testids = folds[f]
        test = [(mi, hi) for mi, hi, pid in trip if pid in testids]
        test_mpos = [mi for mi, hi, pid in trip if pid in testids]
        test_hpos = [hi for mi, hi, pid in trip if pid in testids]
        # clear supervision for test anchors, symmetrically by pair_id (keeps L/R counts matched)
        gm = g0m.copy(); gh = g0h.copy()
        gm[np.isin(apid_m, list(testids))] = False
        gh[np.isin(apid_h, list(testids))] = False
        M.var["garin_anchor"] = gm; H.var["garin_anchor"] = gh
        # train anchor coords for the warp
        train = [(mi, hi) for mi, hi, pid in trip if pid not in testids]
        src = mxyz[[a for a, b in train]]; dst = hxyz[[b for a, b in train]]
        piW = fit(warped_M(src, dst, mxyz, hxyz))
        piE = fit(eucl_M(M.var, H.var))
        mw, dw = score_heldout(piW, test, mxyz, hxyz)
        me, de = score_heldout(piE, test, mxyz, hxyz)
        out[f"fold_{f}"] = {"n_test_anchors": len(test),
                            "warp_mass10mm": mw, "eucl_mass10mm": me,
                            "warp_argmax_dist_mm": dw, "eucl_argmax_dist_mm": de}
        print(f"fold {f}: n={len(test):2d}  mass@10mm warp {mw:.3f} vs eucl {me:.3f}   "
              f"argmax-dist warp {dw:.1f} vs eucl {de:.1f} mm")
        M.var["garin_anchor"] = g0m.copy(); H.var["garin_anchor"] = g0h.copy()
        LOG.write_text(json.dumps(out, indent=2))

    folds_done = [v for k, v in out.items() if k.startswith("fold_")]
    if len(folds_done) == K:
        mw = np.mean([v["warp_mass10mm"] for v in folds_done]); me = np.mean([v["eucl_mass10mm"] for v in folds_done])
        dw = np.mean([v["warp_argmax_dist_mm"] for v in folds_done]); de = np.mean([v["eucl_argmax_dist_mm"] for v in folds_done])
        out["__aggregate__"] = {"warp_mass10mm": float(mw), "eucl_mass10mm": float(me),
                                "warp_argmax_dist_mm": float(dw), "eucl_argmax_dist_mm": float(de)}
        print(f"\nAGGREGATE: mass@10mm warp {mw:.3f} vs eucl {me:.3f};  "
              f"argmax-dist warp {dw:.1f} vs eucl {de:.1f} mm")
        LOG.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    b = int(sys.argv[2]) if len(sys.argv) > 2 else K
    main(a, b)
