#!/usr/bin/env python3

"""CANONICAL Beauchamp metric battery (production scoring + LORO), matching
beauchamp_battery.py but on the canonical coupling recipe.

Modes:
  (default)  score the CANONICAL frozen coupling load_pi() (pi_canonical.npy),
             no refit.  -> outputs/logs/beauchamp_metric_battery_canonical.json
  --loro     leave-one-region-out with WARPED spatial term + eps=0.05,
             xyz_weight=0.25.  For each held-out region the warp is rebuilt
             EXCLUDING that region's Garin pids (no leakage). Resumable.
             -> outputs/logs/beauchamp_metric_battery_loro_canonical.json
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
BBspec = importlib.util.spec_from_file_location("bb", Path(__file__).resolve().parent / "beauchamp_battery.py")
BB = importlib.util.module_from_spec(BBspec); BBspec.loader.exec_module(BB)
from otter.data import load_cached, load_pi
from otter.data.anchors import get_anchor_index
EPS = 0.05
XYZW = 0.25


def warped_M(M, H, exclude_pids=frozenset()):
    im = get_anchor_index(M.var); ih = get_anchor_index(H.var)
    hl = {(int(p), str(h)): int(k) for k, p, h in zip(ih.pos, ih.pair_ids, ih.hemispheres)}
    trip = [(int(mp), hl[(int(pid), str(hm))]) for mp, pid, hm in zip(im.pos, im.pair_ids, im.hemispheres)
            if (int(pid), str(hm)) in hl and int(pid) not in exclude_pids]
    mx = M.var[["x", "y", "z"]].to_numpy(float); hx = H.var[["x", "y", "z"]].to_numpy(float)
    warp = RBFInterpolator(mx[[a for a, b in trip]], hx[[b for a, b in trip]],
                           kernel="thin_plate_spline", smoothing=1e-3)
    d = np.sqrt(((warp(mx)[:, None, :] - hx[None, :, :]) ** 2).sum(-1))
    return (d / max(d.max(), 1e-9)).astype(np.float64)


def main():
    loro = "--loro" in sys.argv
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    pairs, reg_cents, reg_masks, h_xyz, brain_c, pdsq = BB.build(M, H)

    if not loro:
        pi = load_pi().astype(np.float64)
        res = BB.score_all(pi, pairs, reg_cents, reg_masks, h_xyz, brain_c)
        w = np.array([res["per_region"][k]["n_mouse"] for k in res["per_region"]])
        print("PRODUCTION (canonical pi) AGG:", {k: round(x, 3) for k, x in res["aggregate"].items()})
        (ROOT / "outputs/logs/beauchamp_metric_battery_canonical.json").write_text(json.dumps(res, indent=2, default=float))
        print("wrote beauchamp_metric_battery_canonical.json")
        return

    import time
    from otter.data.anchor_packs import build_default_pack_entries
    from otter.models import MultimodalFGW
    CACHE = ROOT / "outputs/logs/beauchamp_metric_battery_loro_canonical.json"
    GUARD = 34.0; t0 = time.time()
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    entries = build_default_pack_entries(M.var, H.var, atlas_root=str(ROOT))
    g0m = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    g0h = H.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    apm = pd.to_numeric(M.var["anchor_pair_id"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    aph = pd.to_numeric(H.var["anchor_pair_id"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    emi = [np.asarray(e.mouse_indices, int) for e in entries]
    res = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [k for k in pairs if k not in res]
    print(f"LORO canonical: {len(res)}/{len(pairs)} done; remaining {len(todo)}")
    for key in todo:
        if time.time() - t0 > GUARD:
            print("guard -> re-run to continue"); break
        mm, hm = pairs[key]
        pids = set(apm[g0m & mm & np.isfinite(apm)].astype(int).tolist()) or {-999}
        dm = g0m & np.isin(np.nan_to_num(apm, nan=-1).astype(int), list(pids))
        dh = g0h & np.isin(np.nan_to_num(aph, nan=-1).astype(int), list(pids))
        M.var["garin_anchor"] = g0m & ~dm; H.var["garin_anchor"] = g0h & ~dh
        ent_ho = [e for e, mi in zip(entries, emi) if not mm[mi].any()]
        Mxyz = warped_M(M, H, exclude_pids={p for p in pids if p != -999})
        m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7, epsilon=EPS,
                          xyz_weight=XYZW, lam_anchor=1.0, alpha=0.5)
        m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"], region_anchors=ent_ho, M_xyz=Mxyz)
        M.var["garin_anchor"] = g0m; H.var["garin_anchor"] = g0h
        res[key] = BB.battery(m.pi.astype(np.float64), mm, hm, h_xyz, reg_cents, reg_masks, key, brain_c)
        CACHE.write_text(json.dumps(res, indent=2, default=float))
        v = res[key]
        print(f"  {key.split(' -> ')[0][:20]:20s} top1={v['top1']:.2f} auroc={v['auroc']:.2f} "
              f"disp={v['centroid_disp_mm']:.0f} nr19={v['nr19_hit']}", flush=True)
    if not todo:
        ks = list(res)
        for pkey, qkey in (("perm_p_mass", "perm_q_mass"), ("spin_p_disp", "spin_q_disp")):
            q = BB.bh_fdr([res[k][pkey] for k in ks])
            for k, qq in zip(ks, q): res[k][qkey] = float(qq)
        CACHE.write_text(json.dumps(res, indent=2, default=float))
        w = np.array([res[k]["n_mouse"] for k in ks])
        print("LORO canonical AGG auroc=%.2f disp=%.0f top1=%.2f nr19=%.2f" % (
            np.average([res[k]["auroc"] for k in ks], weights=w),
            np.average([res[k]["centroid_disp_mm"] for k in ks], weights=w),
            np.average([res[k]["top1"] for k in ks], weights=w),
            np.mean([res[k]["nr19_hit"] for k in ks])))
        print("ALL DONE")


if __name__ == "__main__":
    main()
