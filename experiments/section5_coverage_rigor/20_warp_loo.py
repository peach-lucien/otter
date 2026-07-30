#!/usr/bin/env python3
"""Circularity control for the anchor-driven warp: leave-one-anchor-out generalization.

The TPS warp is fit on 42 Garin homolog anchors and improves exactly the Beauchamp pairs that
overlap those anchors. Is that generalization (the warp learned the global mouse->human deformation)
or memorization (it pinned the specific anchor)? Test: for each anchor-overlapping Beauchamp pair,
DROP the anchor control point(s) whose mouse position lies in the pair's mouse mask (or human in the
human mask), refit the warp WITHOUT them, refit the coupling, and re-score THAT pair held-out. If the
held-out top1 still beats the Euclidean baseline, the warp generalizes.

Runs a slice of pairs (argv start end) and appends to outputs/logs/section5_warp_loo.json so it can
be chunked under the bash timeout. ~5s per pair (one coupling refit).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments/section5_coverage_rigor"))
from homer.data import load_cached                       # noqa: E402
from homer.data.anchors import get_anchor_index          # noqa: E402
from homer.models import MultimodalFGW                   # noqa: E402
from beauchamp_scorer import BeauchampScorer             # noqa: E402

LOG = ROOT / "outputs/logs/section5_warp_loo.json"


def build_M_xyz(warped_mouse, hxyz):
    d = np.sqrt(((warped_mouse[:, None, :] - hxyz[None, :, :]) ** 2).sum(-1))
    return (d / max(d.max(), 1e-9)).astype(np.float64)


def main(start, end):
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    mxyz = M.var[["x", "y", "z"]].to_numpy(float); hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    im = get_anchor_index(M.var); ih = get_anchor_index(H.var)
    hpos = {int(p): k for k, p in zip(ih.pos, ih.pair_ids)}
    pairs = [(int(mp), hpos[int(pid)]) for mp, pid in zip(im.pos, im.pair_ids) if int(pid) in hpos]
    a_mpos = np.array([a for a, b in pairs]); a_hpos = np.array([b for a, b in pairs])
    src_all = mxyz[a_mpos]; dst_all = hxyz[a_hpos]

    sc = BeauchampScorer()
    euc = np.load("/var/tmp/pi_warp_euclid_baseline.npy")
    re = sc.score(euc)
    # full-warp scores for reference
    tps = np.load("/var/tmp/pi_warp_tps.npy"); rt = sc.score(tps)

    def fit_and_score(keep_mask, pairkey):
        warp = RBFInterpolator(src_all[keep_mask], dst_all[keep_mask], kernel="thin_plate_spline", smoothing=1e-3)
        Mx = build_M_xyz(warp(mxyz), hxyz)
        m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7, epsilon=5e-3,
                          xyz_weight=0.5, lam_anchor=1.0, alpha=0.5)
        m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"], M_xyz=Mx, region_anchors=[])
        return sc.evaluate(m.pi.astype(np.float64), sc.m_masks[pairkey], sc.h_masks[pairkey])

    scorable = [p for p in sc.pairs if sc.is_anchor[p] and p in re and p in rt]
    out = json.loads(LOG.read_text()) if LOG.exists() else {}
    for p in scorable[start:end]:
        mm = sc.m_masks[p]; hm = sc.h_masks[p]
        # anchors whose mouse pos is in this pair's mouse mask OR human pos in human mask
        drop = np.array([mm[a_mpos[i]] or hm[a_hpos[i]] for i in range(len(pairs))])
        keep = ~drop
        base = re[p]["top1"]; full = rt[p]["top1"]
        if drop.sum() == 0 or keep.sum() < 6:
            out[p] = {"n_anchors_dropped": int(drop.sum()), "euclid_top1": base, "fullwarp_top1": full,
                      "loo_top1": None, "note": "no matching anchor or too few remain"}
        else:
            held = fit_and_score(keep, p)
            out[p] = {"n_anchors_dropped": int(drop.sum()), "euclid_top1": base,
                      "fullwarp_top1": full, "loo_top1": float(held["top1"])}
        print(f"{p:<46} drop={out[p]['n_anchors_dropped']}  euclid={base:.2f} "
              f"full={full:.2f} LOO={out[p]['loo_top1'] if out[p]['loo_top1'] is None else round(out[p]['loo_top1'],2)}")
        LOG.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    e = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    main(s, e)
