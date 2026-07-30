#!/usr/bin/env python3
"""Region-anchor hold-out CV on the CANONICAL model (warp + xyz_weight=0.25 + eps=0.05).
Reuses 05h's helpers; injects the warped M_xyz + canonical hyperparams. Resumable per pid.
Writes outputs/logs/region_anchor_cv_canonical.json (same schema as region_anchor_cv.json).
Run in chunks: python 36_region_anchor_cv_canonical.py [start] [end]
"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("h05", ROOT / "pipeline/05h_region_anchor_cv.py")
H05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(H05)
from homer.data import load_cached                          # noqa: E402
from homer.data.anchors import get_anchor_index             # noqa: E402
from homer.data.atlas_regions import build_garin_region_anchors_from_atlases  # noqa: E402
from homer.models import MultimodalFGW                      # noqa: E402

OUT = ROOT / "outputs/logs/region_anchor_cv_canonical.json"
EPS, XYZW = 0.05, 0.25


def warped_M(M, H):
    im = get_anchor_index(M.var); ih = get_anchor_index(H.var)
    hl = {(int(p), str(h)): int(k) for k, p, h in zip(ih.pos, ih.pair_ids, ih.hemispheres)}
    trip = [(int(mp), hl[(int(pid), str(hm))]) for mp, pid, hm in zip(im.pos, im.pair_ids, im.hemispheres)
            if (int(pid), str(hm)) in hl]
    mx = M.var[["x", "y", "z"]].to_numpy(float); hx = H.var[["x", "y", "z"]].to_numpy(float)
    warp = RBFInterpolator(mx[[a for a, b in trip]], hx[[b for a, b in trip]], kernel="thin_plate_spline", smoothing=1e-3)
    d = np.sqrt(((warp(mx)[:, None, :] - hx[None, :, :]) ** 2).sum(-1))
    return (d / max(d.max(), 1e-9)).astype(np.float64)


def main(start, end):
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    Mxyz = warped_M(M, H)
    entries = build_garin_region_anchors_from_atlases(M.var, H.var)
    n_h = len(H.var)
    state = json.loads(OUT.read_text()) if OUT.exists() else {}

    def fit(visible):
        m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7, epsilon=EPS, xyz_weight=XYZW, lam_anchor=1.0)
        m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"], region_anchors=visible, M_xyz=Mxyz)
        return m

    if "_full_train" not in state:
        t = time.time(); m = fit(entries)
        state["_full_train"] = {"per_region": {str(e.pair_id): H05.evaluate_region_recovery(
            m.pi, e.mouse_indices, e.human_indices, n_h) for e in entries}, "elapsed_s": round(time.time()-t, 1)}
        OUT.write_text(json.dumps(state, indent=2, default=float)); print(f"full-train {state['_full_train']['elapsed_s']}s")

    ho = state.setdefault("_held_out", {})
    for i, held in enumerate(entries[start:end], start=start):
        key = str(held.pair_id)
        if key in ho:
            print(f"  {i+1}/{len(entries)} pid={held.pair_id} cached"); continue
        t = time.time()
        m = fit([e for e in entries if e.pair_id != held.pair_id])
        r = H05.evaluate_region_recovery(m.pi, held.mouse_indices, held.human_indices, n_h)
        r["elapsed_s"] = round(time.time()-t, 1); r["label"] = held.label[:80]
        ho[key] = r; OUT.write_text(json.dumps(state, indent=2, default=float))
        print(f"  {i+1}/{len(entries)} pid={held.pair_id} {held.label[:34]:34s} top1={r['top1']:.0%} ({r['elapsed_s']}s)", flush=True)
    done = len(ho)
    print(f"held-out done {done}/{len(entries)}")
    if done == len(entries):
        w = np.array([ho[str(e.pair_id)]["n_mouse"] for e in entries if not np.isnan(ho[str(e.pair_id)]["top1"])], float)
        t1 = np.array([ho[str(e.pair_id)]["top1"] for e in entries if not np.isnan(ho[str(e.pair_id)]["top1"])])
        print(f"WEIGHTED held-out top1 = {(w*t1).sum()/w.sum():.1%}")


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    e = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    main(s, e)
