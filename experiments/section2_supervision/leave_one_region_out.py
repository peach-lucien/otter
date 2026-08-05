#!/usr/bin/env python3
"""Leave-one-region-out generalisation test of the FULL OTTER model.

For each Beauchamp pair, remove ALL curated supervision that anchors that
region -- its Garin point anchor(s) AND any region-anchor pack whose mouse
parcels fall in the region -- re-fit the full FGW coupling with everything
else, and score that pair's Beauchamp recovery. This is the honest
"would we recover this region if we had NOT curated it" number for the
production model (Garin + all packs), unlike 05h which is Garin-only.

Resumable: caches per-pair to .loro_results.json; exits after TIME_GUARD s so
it fits the sandbox's per-call limit. Re-run until 'ALL DONE'.
"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path("/sessions/modest-tender-carson/mnt/brain_crossspecies_translation/otter")
OUT = Path("/sessions/modest-tender-carson/mnt/outputs")
CACHE = OUT / ".loro_results_v2.json"
TIME_GUARD = 34.0
t_start = time.time()

sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached
from otter.data.anchor_packs import build_default_pack_entries
from otter.models import MultimodalFGW

# import 05f for masks + scorer
spec = importlib.util.spec_from_file_location("b05f", ROOT / "pipeline/05f_beauchamp_validation.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

def fit_full(M, H, costs, entries):
    m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                      epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
    m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"], region_anchors=entries)
    return m.pi

def region_metrics(pi, m_mask, h_mask, h_xyz):
    """Continuous recovery measures for a region.
       top1        : per-parcel argmax-in-region hit-rate (the harsh metric)
       mass        : fraction of the region's routed mass landing in the true region
       cdist_mm    : distance (mm) between mass-weighted predicted human centroid
                     and the true human region centroid  (the 'MSE-like' error)
       amdist_mm   : mean per-parcel distance from argmax human parcel to true centroid
       rand_mm     : expected distance if routing were random (null baseline)
    """
    hidx = np.where(h_mask)[0]
    true_c = h_xyz[hidx].mean(0)
    block = pi[m_mask]
    tot = block.sum(0)
    s = tot.sum()
    if s <= 0:
        return dict(top1=0.0, mass=0.0, cdist_mm=float("nan"), amdist_mm=float("nan"))
    distn = tot / s
    am = block.argmax(1)
    top1 = float(np.isin(am, hidx).mean())
    mass = float(tot[h_mask].sum() / s)
    pred_c = (distn[:, None] * h_xyz).sum(0)
    cdist = float(np.linalg.norm(pred_c - true_c))
    amdist = float(np.linalg.norm(h_xyz[am] - true_c[None, :], axis=1).mean())
    rand_mm = float(np.linalg.norm(h_xyz - true_c[None, :], axis=1).mean())
    return dict(top1=top1, mass=mass, cdist_mm=cdist, amdist_mm=amdist, rand_mm=rand_mm)

def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    entries = build_default_pack_entries(M.var, H.var, atlas_root=str(ROOT))

    # masks (once)
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

    gcol0_m = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    gcol0_h = H.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    apid_m = M.var["anchor_pair_id"].to_numpy()
    apid_h = H.var["anchor_pair_id"].to_numpy()
    ent_mouse_idx = [np.asarray(e.mouse_indices, dtype=int) for e in entries]

    res = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    # full-model baseline (once)
    if "__full__" not in res:
        pi = fit_full(M, H, costs, entries)
        full = {}
        for key, (m_mask, h_mask) in pairs.items():
            full[key] = region_metrics(pi, m_mask, h_mask, h_xyz)
        res["__full__"] = full
        CACHE.write_text(json.dumps(res, indent=2, default=float))
        print(f"[full] fitted+scored {len(full)} pairs ({time.time()-t_start:.1f}s)")

    todo = [k for k in pairs if k not in res]
    print(f"remaining LORO pairs: {len(todo)}")
    for key in todo:
        if time.time() - t_start > TIME_GUARD:
            print("time guard -> exit, re-run to continue"); return
        m_mask, h_mask = pairs[key]
        # garin pids located in this region (mouse side)
        finite = np.isfinite(apid_m)
        region_pids = set(apid_m[gcol0_m & m_mask & finite].astype(int).tolist())
        # hold out: unflag garin for those pids on BOTH species
        drop_m = gcol0_m & np.isin(np.nan_to_num(apid_m, nan=-1).astype(int), list(region_pids) or [-999])
        drop_h = gcol0_h & np.isin(np.nan_to_num(apid_h, nan=-1).astype(int), list(region_pids) or [-999])
        M.var["garin_anchor"] = gcol0_m & ~drop_m
        H.var["garin_anchor"] = gcol0_h & ~drop_h
        # drop packs overlapping the region
        ent_ho = [e for e, mi in zip(entries, ent_mouse_idx) if not m_mask[mi].any()]
        t0 = time.time()
        pi = fit_full(M, H, costs, ent_ho)
        ho = region_metrics(pi, m_mask, h_mask, h_xyz)
        # restore var
        M.var["garin_anchor"] = gcol0_m
        H.var["garin_anchor"] = gcol0_h
        fm = res["__full__"][key]
        res[key] = {
            "full": fm, "heldout": ho,
            "n_mouse": int(m_mask.sum()),
            "n_garin_pids_removed": len(region_pids),
            "n_pack_entries_removed": len(entries) - len(ent_ho),
        }
        CACHE.write_text(json.dumps(res, indent=2, default=float))
        print(f"  {key.split(' -> ')[0][:26]:26s} top1 {fm['top1']:.2f}>{ho['top1']:.2f}  "
              f"mass {fm['mass']:.2f}>{ho['mass']:.2f}  "
              f"cdist {fm['cdist_mm']:.0f}>{ho['cdist_mm']:.0f}mm  rand {ho['rand_mm']:.0f}mm  {time.time()-t0:.1f}s")

    if not todo:
        print("ALL DONE")

if __name__ == "__main__":
    main()
