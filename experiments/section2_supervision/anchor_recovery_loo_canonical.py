#!/usr/bin/env python3

"""CANONICAL VALIDATION 1: anchor-recovery leave-one-out on the canonical model.

Same 41-unit leave-one-out as anchor_recovery_loo.py (15 Garin classes + 26 packs),
but refit with the CANONICAL coupling recipe:
  - WARPED spatial term (thin-plate-spline RBF over Garin homolog coord pairs)
  - epsilon = 0.05, xyz_weight = 0.25

The warp is trained from Garin anchor coord-pairs. For each held-out unit the warp is
rebuilt EXCLUDING the held-out pair_ids (the same pids whose garin_anchor flags are
unflagged), so no held-out spatial correspondence leaks into the refit. For pack units
with no Garin anchor inside them the warp is the full canonical warp.

Writes outputs/logs/anchor_recovery_loo_combined_canonical.json.
Resumable (34 s guard). Re-run until 'ALL DONE'.
"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
BBspec = importlib.util.spec_from_file_location("bb", Path(__file__).resolve().parent / "beauchamp_battery.py")
BB = importlib.util.module_from_spec(BBspec); BBspec.loader.exec_module(BB)
from otter.data import load_cached
from otter.data.anchors import get_anchor_index
from otter.data.anchor_packs import build_default_pack_entries
from otter.data.atlas_regions import build_garin_region_anchors_from_atlases
from otter.models import MultimodalFGW

CACHE = ROOT / "outputs/logs/anchor_recovery_loo_combined_canonical.json"
GUARD = 34.0
EPS = 0.05
XYZW = 0.25


def warped_M(M, H, exclude_pids=frozenset()):
    """Thin-plate-spline warp of mouse xyz -> human xyz over Garin homolog pairs,
    EXCLUDING any pair_id in exclude_pids (prevents held-out leakage). Returns
    normalized cross-species spatial cost matrix (mouse x human)."""
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
    t0 = time.time()
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    entries = build_default_pack_entries(M.var, H.var, atlas_root=str(ROOT))
    garin_regs = build_garin_region_anchors_from_atlases(M.var, H.var)
    nH = len(H.var); h_xyz = H.var[["x", "y", "z"]].to_numpy(); brain_c = h_xyz.mean(0)

    def mm_of(e): b = np.zeros(len(M.var), bool); b[list(e.mouse_indices)] = True; return b
    def hm_of(e): b = np.zeros(nH, bool); b[list(e.human_indices)] = True; return b
    units = []
    for e in garin_regs:
        mm, hm = mm_of(e), hm_of(e)
        if mm.sum() and hm.sum(): units.append((f"Garin:{getattr(e,'label','')}"[:36], mm, hm, "garin"))
    for e in entries:
        mm, hm = mm_of(e), hm_of(e)
        if mm.sum() and hm.sum(): units.append((f"Pack:{getattr(e,'label','')}"[:36], mm, hm, "pack"))
    reg_cents = {k: h_xyz[np.where(hm)[0]].mean(0) for k, mm, hm, kind in units}
    reg_masks = {k: hm for k, mm, hm, kind in units}
    keys = [u[0] for u in units]

    g0m = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    g0h = H.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    apm = pd.to_numeric(M.var["anchor_pair_id"], errors="coerce").to_numpy("float64", na_value=np.nan)
    aph = pd.to_numeric(H.var["anchor_pair_id"], errors="coerce").to_numpy("float64", na_value=np.nan)
    emi = [np.asarray(e.mouse_indices, int) for e in entries]

    res = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [k for k in keys if k not in res]
    print(f"CANONICAL anchor-recovery LOO ({len(keys)} units = {sum(u[3]=='garin' for u in units)} Garin + "
          f"{sum(u[3]=='pack' for u in units)} packs): {len(res)} done; remaining {len(todo)}")
    for key, mm, hm, kind in units:
        if key in res: continue
        if time.time() - t0 > GUARD:
            print("guard -> re-run to continue"); break
        pids = set(apm[g0m & mm & np.isfinite(apm)].astype(int).tolist()) or {-999}
        dm = g0m & np.isin(np.nan_to_num(apm, nan=-1).astype(int), list(pids))
        dh = g0h & np.isin(np.nan_to_num(aph, nan=-1).astype(int), list(pids))
        M.var["garin_anchor"] = g0m & ~dm; H.var["garin_anchor"] = g0h & ~dh
        keep = [ent for ent, mi in zip(entries, emi) if (mm[mi].mean() if len(mi) else 0.0) <= 0.5]
        # canonical warp, excluding held-out pids (no spatial leakage of the held-out unit)
        Mxyz = warped_M(M, H, exclude_pids={p for p in pids if p != -999})
        m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7, epsilon=EPS,
                          xyz_weight=XYZW, lam_anchor=1.0, alpha=0.5)
        m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"], region_anchors=keep, M_xyz=Mxyz)
        M.var["garin_anchor"] = g0m; H.var["garin_anchor"] = g0h
        res[key] = BB.battery(m.pi.astype(np.float64), mm, hm, h_xyz, reg_cents, reg_masks, key, brain_c)
        res[key]["kind"] = kind
        res[key]["n_garin_pids_removed"] = len(pids if pids != {-999} else [])
        res[key]["n_packs_removed"] = len(entries) - len(keep)
        CACHE.write_text(json.dumps(res, indent=2, default=float))
        v = res[key]
        print(f"  {key[:34]:34s} auroc={v['auroc']:.2f} top1={v['top1']:.3f} disp={v['centroid_disp_mm']:.0f} "
              f"(-{v['n_garin_pids_removed']}g/-{v['n_packs_removed']}pk) {time.time()-t0:.0f}s", flush=True)
    if not todo:
        ks = list(res)
        for pk, qk in (("perm_p_mass", "perm_q_mass"), ("spin_p_disp", "spin_q_disp")):
            q = BB.bh_fdr([res[k][pk] for k in ks])
            for k, qq in zip(ks, q): res[k][qk] = float(qq)
        CACHE.write_text(json.dumps(res, indent=2, default=float))
        def agg(sub):
            if not sub: return "n/a"
            ww = np.array([res[k]["n_mouse"] for k in sub])
            return "AUROC=%.3f disp=%.0fmm top1=%.4f (n=%d)" % (
                np.average([res[k]["auroc"] for k in sub], weights=ww),
                np.average([res[k]["centroid_disp_mm"] for k in sub], weights=ww),
                np.average([res[k]["top1"] for k in sub], weights=ww), len(sub))
        gk = [k for k in ks if res[k].get("kind") == "garin"]; pkk = [k for k in ks if res[k].get("kind") == "pack"]
        print("COMBINED:", agg(ks)); print("  Garin regions:", agg(gk)); print("  Packs:", agg(pkk))
        print("ALL DONE")


if __name__ == "__main__":
    main()
