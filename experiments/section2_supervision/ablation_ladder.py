#!/usr/bin/env python3
"""Recompute the supervision build-up ladder with the FULL metric battery.

Four stages, each a fresh fit, scored with AUROC / displacement / mass / top-k
(aggregate over the 19 Beauchamp regions):
  connectivity : FC+SC relational only (no feature term: xyz=0, no anchors/packs)
  + spatial    : + xyz
  + anchors    : + 21 Garin point anchors
  + packs      : + region-anchor packs   (= production)

Resumable per stage. -> outputs/logs/ablation_ladder_battery.json
Run: cd otter && PYTHONPATH=src python ../manuscript/figures/fig_2_ED/ablation_ladder.py
"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
bb = importlib.util.spec_from_file_location("bb", Path(__file__).resolve().parent / "beauchamp_battery.py")
BB = importlib.util.module_from_spec(bb); bb.loader.exec_module(BB)
from otter.data import load_cached
from otter.data.anchor_packs import build_default_pack_entries
from otter.models import MultimodalFGW

CACHE = ROOT / "outputs/logs/ablation_ladder_battery.json"
STAGES = ["connectivity", "+spatial", "+anchors", "+packs"]

def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    entries = build_default_pack_entries(M.var, H.var, atlas_root=str(ROOT))
    pairs, reg_cents, reg_masks, h_xyz, brain_c, pdsq = BB.build(M, H)
    g0m = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    g0h = H.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    res = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def fit(xyz, garin, packs):
        M.var["garin_anchor"] = g0m if garin else (g0m & False)
        H.var["garin_anchor"] = g0h if garin else (g0h & False)
        m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7, epsilon=5e-3,
                          xyz_weight=xyz, lam_anchor=1.0, alpha=0.5)
        m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
              region_anchors=(entries if packs else []))
        M.var["garin_anchor"] = g0m; H.var["garin_anchor"] = g0h
        return m.pi.astype(np.float64)

    cfg = {"connectivity": (0.0, False, False), "+spatial": (0.5, False, False),
           "+anchors": (0.5, True, False), "+packs": (0.5, True, True)}
    for st in STAGES:
        if st in res:
            print(f"  {st}: cached"); continue
        t = time.time(); pi = fit(*cfg[st])
        agg = BB.score_all(pi, pairs, reg_cents, reg_masks, h_xyz, brain_c)["aggregate"]
        res[st] = agg; CACHE.write_text(json.dumps(res, indent=2, default=float))
        print(f"  {st:14s} top1={agg['top1']:.2f} top5={agg['top5']:.2f} top10={agg['top10']:.2f} "
              f"AUROC={agg['auroc']:.2f} mass={agg['mass_in_region']:.2f} disp={agg['centroid_disp_mm']:.0f}mm "
              f"({time.time()-t:.0f}s)", flush=True)
    print("stages done:", len([s for s in STAGES if s in res]), "/4")

if __name__ == "__main__":
    main()
