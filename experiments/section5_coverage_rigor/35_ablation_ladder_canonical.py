#!/usr/bin/env python3

"""Section 2 ablation ladder, recomputed on the CANONICAL model (warp + xyz_weight=0.25 + eps=0.05).
Four stages, fresh fit each, scored with the Beauchamp metric battery:
  connectivity : FC+SC only (xyz_weight=0, no anchors/packs)
  + spatial    : + warped xyz (xyz_weight=0.25)
  + anchors    : + 21 Garin point anchors
  + packs      : + region packs (= canonical pi)
Writes outputs/logs/ablation_ladder_battery_canonical.json
"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
BBP = ROOT.parent / "manuscript/figures/fig_2_ED/beauchamp_battery.py"
spec = importlib.util.spec_from_file_location("bb", BBP); BB = importlib.util.module_from_spec(spec); spec.loader.exec_module(BB)
from otter.data import load_cached                          # noqa: E402
from otter.data.anchors import get_anchor_index             # noqa: E402
from otter.data.anchor_packs import build_default_pack_entries   # noqa: E402
from otter.models import MultimodalFGW                      # noqa: E402

OUT = ROOT / "outputs/logs/ablation_ladder_battery_canonical.json"
STAGES = ["connectivity", "+spatial", "+anchors", "+packs"]
EPS = 0.05
XYZW = 0.25


def warped_M(M, H):
    im = get_anchor_index(M.var); ih = get_anchor_index(H.var)
    hl = {(int(p), str(h)): int(k) for k, p, h in zip(ih.pos, ih.pair_ids, ih.hemispheres)}
    trip = [(int(mp), hl[(int(pid), str(hm))]) for mp, pid, hm in zip(im.pos, im.pair_ids, im.hemispheres)
            if (int(pid), str(hm)) in hl]
    mx = M.var[["x", "y", "z"]].to_numpy(float); hx = H.var[["x", "y", "z"]].to_numpy(float)
    warp = RBFInterpolator(mx[[a for a, b in trip]], hx[[b for a, b in trip]], kernel="thin_plate_spline", smoothing=1e-3)
    d = np.sqrt(((warp(mx)[:, None, :] - hx[None, :, :]) ** 2).sum(-1))
    return (d / max(d.max(), 1e-9)).astype(np.float64)


def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    entries = build_default_pack_entries(M.var, H.var, atlas_root=str(ROOT))
    Mxyz = warped_M(M, H)
    pairs, reg_cents, reg_masks, h_xyz, brain_c, pdsq = BB.build(M, H)
    g0m = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    g0h = H.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}

    def fit(xyzw, garin, packs):
        M.var["garin_anchor"] = g0m if garin else (g0m & False)
        H.var["garin_anchor"] = g0h if garin else (g0h & False)
        m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7, epsilon=EPS,
                          xyz_weight=xyzw, lam_anchor=1.0, alpha=0.5)
        kw = {} if xyzw == 0 else {"M_xyz": Mxyz}
        m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
              region_anchors=(entries if packs else []), **kw)
        M.var["garin_anchor"] = g0m; H.var["garin_anchor"] = g0h
        return m.pi.astype(np.float64)

    cfg = {"connectivity": (0.0, False, False), "+spatial": (XYZW, False, False),
           "+anchors": (XYZW, True, False), "+packs": (XYZW, True, True)}
    for st in STAGES:
        if st in res:
            print(f"  {st}: cached"); continue
        t = time.time(); pi = fit(*cfg[st])
        agg = BB.score_all(pi, pairs, reg_cents, reg_masks, h_xyz, brain_c)["aggregate"]
        res[st] = agg; OUT.write_text(json.dumps(res, indent=2, default=float))
        print(f"  {st:14s} top1={agg['top1']:.2f} AUROC={agg['auroc']:.2f} "
              f"mass={agg['mass_in_region']:.2f} disp={agg['centroid_disp_mm']:.0f}mm ({time.time()-t:.0f}s)", flush=True)
    print("done", OUT)


if __name__ == "__main__":
    main()
