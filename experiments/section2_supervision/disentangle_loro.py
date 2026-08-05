#!/usr/bin/env python3
"""TIER-1 control: does connectivity localise a held-out region BEYOND spatial position?

For each Beauchamp region, remove its curated supervision (Garin anchor + overlapping
packs) and re-fit the full model under three configs, then score localisation:
  both      : production  (alpha=0.5 GW + xyz=0.5)          -> connectivity + space
  xyz_only  : alpha=0     (no GW/connectivity) + xyz=0.5     -> space (+ other anchors)
  conn_only : alpha=0.5 GW + xyz=0                           -> connectivity (+ other anchors)

If both ≈ xyz_only, connectivity adds nothing to localisation. If both < xyz_only,
connectivity helps. conn_only shows connectivity acting alone.
Resumable; caches to .disentangle_v1.json.
"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path("/sessions/modest-tender-carson/mnt/brain_crossspecies_translation/otter")
OUT = Path("/sessions/modest-tender-carson/mnt/outputs")
CACHE = OUT / ".disentangle_v1.json"
TIME_GUARD = 33.0
t_start = time.time()

sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached
from otter.data.anchor_packs import build_default_pack_entries
from otter.models import MultimodalFGW

spec = importlib.util.spec_from_file_location("b05f", ROOT / "pipeline/05f_beauchamp_validation.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

CONFIGS = {
    "both":      dict(alpha=0.5, xyz_weight=0.5),
    "xyz_only":  dict(alpha=0.0, xyz_weight=0.5),
    "conn_only": dict(alpha=0.5, xyz_weight=0.0),
}

def fit(M, H, costs, entries, cfg):
    m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7, epsilon=5e-3,
                      lam_anchor=1.0, alpha=cfg["alpha"], xyz_weight=cfg["xyz_weight"])
    m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"], region_anchors=entries)
    return m.pi

def loc(pi, m_mask, h_mask, h_xyz):
    hidx = np.where(h_mask)[0]; true_c = h_xyz[hidx].mean(0)
    block = pi[m_mask]; tot = block.sum(0); s = tot.sum()
    if s <= 0: return dict(cdist_mm=float("nan"), mass=0.0, top1=0.0)
    distn = tot / s
    pred_c = (distn[:, None] * h_xyz).sum(0)
    return dict(
        cdist_mm=float(np.linalg.norm(pred_c - true_c)),
        mass=float(tot[h_mask].sum() / s),
        top1=float(np.isin(block.argmax(1), hidx).mean()),
        rand_mm=float(np.linalg.norm(h_xyz - true_c[None, :], axis=1).mean()),
    )

def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    entries = build_default_pack_entries(M.var, H.var, atlas_root=str(ROOT))
    name2lab = b.parse_dsurqe_tree(b.EXT / "AMBA/data/DSURQE_tree.json")
    parcel_dsurqe = b.assign_dsurqe_labels(M, b.EXT / "AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc")
    human_mem = b.assign_human_region_membership(H, b.HUMAN_REGION_MNI)
    h_xyz = H.var[["x", "y", "z"]].to_numpy()

    pairs = {}
    for mname, hname in b.BEAUCHAMP_PAIRS:
        if mname not in name2lab: continue
        m_mask = np.isin(parcel_dsurqe, list(name2lab[mname]))
        h_mask = human_mem.get(hname)
        if h_mask is None or m_mask.sum() == 0 or h_mask.sum() == 0: continue
        pairs[f"{mname} -> {hname}"] = (m_mask, h_mask)

    g0m = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    g0h = H.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    apid_m = M.var["anchor_pair_id"].to_numpy(); apid_h = H.var["anchor_pair_id"].to_numpy()
    emi = [np.asarray(e.mouse_indices, dtype=int) for e in entries]

    res = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [k for k in pairs if k not in res]
    print(f"remaining: {len(todo)}")
    for key in todo:
        if time.time() - t_start > TIME_GUARD:
            print("time guard -> exit, re-run"); return
        m_mask, h_mask = pairs[key]
        fin = np.isfinite(apid_m)
        pids = set(apid_m[g0m & m_mask & fin].astype(int).tolist()) or {-999}
        dm = g0m & np.isin(np.nan_to_num(apid_m, nan=-1).astype(int), list(pids))
        dh = g0h & np.isin(np.nan_to_num(apid_h, nan=-1).astype(int), list(pids))
        M.var["garin_anchor"] = g0m & ~dm; H.var["garin_anchor"] = g0h & ~dh
        ent_ho = [e for e, mi in zip(entries, emi) if not m_mask[mi].any()]
        row = {}
        for cname, cfg in CONFIGS.items():
            pi = fit(M, H, costs, ent_ho, cfg)
            row[cname] = loc(pi, m_mask, h_mask, h_xyz)
        M.var["garin_anchor"] = g0m; H.var["garin_anchor"] = g0h
        res[key] = row; CACHE.write_text(json.dumps(res, indent=2, default=float))
        print(f"  {key.split(' -> ')[0][:24]:24s} cdist both={row['both']['cdist_mm']:.0f} "
              f"xyz={row['xyz_only']['cdist_mm']:.0f} conn={row['conn_only']['cdist_mm']:.0f} "
              f"rand={row['both']['rand_mm']:.0f}mm")
    if not todo: print("ALL DONE")

if __name__ == "__main__":
    main()
