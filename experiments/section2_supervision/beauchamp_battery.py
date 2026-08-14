#!/usr/bin/env python3
"""Full metric battery + nulls for the Beauchamp benchmark.

RETIRED ENTRY POINT — run `beauchamp_battery_canonical.py` instead.
    This module is still LIVE as a *library*: `beauchamp_battery_canonical.py`
    imports its `build()` and `score_all()`, and so do the other live scoring
    scripts. What is retired is its own `main()`, which
    scored the pre-canonical coupling and wrote
    `outputs/logs/beauchamp_metric_battery.json` /
    `beauchamp_metric_battery_loro.json`. Every live consumer now reads the
    `*_canonical.json` logs written by `beauchamp_battery_canonical.py`.

    The hardcoded pre-warp coupling `pi_fc_plus_SC_with_all_packs` in `main()` was repointed to
    canonical `load_pi()` on 2026-07-18 so the retired coupling cannot re-enter
    through this path. `main()` was NOT re-run and the two non-canonical logs it
    owns were left untouched — running it would merely duplicate
    `beauchamp_battery_canonical.py` under a filename that live scripts do not read.

    NOTE: the Beauchamp results in `docs/03_results.md` still cite
    `beauchamp_metric_battery.json` (+ `_loro.json`); they should cite the
    `_canonical` logs the numbers are actually built from.

Modes:
  (default)  score the canonical coupling via load_pi() (RETIRED entry point; see above),
             no re-fit.  -> outputs/logs/beauchamp_metric_battery.json
  --loro     leave-one-region-out: remove a region's curation (Garin anchor + overlapping
             packs), re-fit the full model, score the held-out region with the SAME
             battery.  Resumable.  -> outputs/logs/beauchamp_metric_battery_loro.json

Battery (per region):
  Rank        top1, top5, top10, mean_rank
  Mass/spread mass_in_region, auroc, auprc, entropy(norm), size_norm
  Distance    centroid_disp_mm, expected_disp_mm, argmax_disp_mm
  Identify    nr19 (nearest of the 19 Beauchamp regions) hit+rank
              nratlas (nearest of the whole-brain H.var 'region' atlas) hit+rank
  Nulls       analytic chance + enrichment (top-k)
              perm_p_mass / perm_p_disp : REGION-LABEL PERMUTATION (primary null) -- rank
                        of the true region's mass / proximity among the 19 real regions
              spin_p_disp : rotation of the true centroid about the brain centre (secondary)
  CI          bootstrap 95% CI (PARCEL-level: resample mouse parcels in the region)
Aggregate is parcel-weighted; BH-FDR applied to perm_p_disp across regions.
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import entropy as shannon
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi
spec = importlib.util.spec_from_file_location("b05f", ROOT / "pipeline/05f_beauchamp_validation.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
rng = np.random.default_rng(0)
NSPIN, NBOOT, NPERM = 1000, 500, 1000

def bh_fdr(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); r = np.empty(n)
    r[o] = (p[o] * n / (np.arange(n) + 1));
    # enforce monotonicity
    q = np.minimum.accumulate(r[o][::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1)
    return out

def battery(pi, m_mask, h_mask, h_xyz, reg_cents, reg_masks, true_key, brain_c):
    hidx = np.where(h_mask)[0]; hset = set(hidx.tolist()); nh = pi.shape[1]
    blk = pi[m_mask]; nm = blk.shape[0]; tot = blk.sum(0); s = tot.sum(); totn = tot / s
    order = np.argsort(-blk, 1)
    top1 = float(np.isin(order[:, 0], hidx).mean())
    top5 = float(np.mean([bool(hset & set(order[i, :5])) for i in range(nm)]))
    top10 = float(np.mean([bool(hset & set(order[i, :10])) for i in range(nm)]))
    ranks = []
    for i in range(nm):
        pos = np.where(np.isin(order[i], hidx))[0]; ranks.append(int(pos[0]) + 1 if len(pos) else nh)
    mean_rank = float(np.mean(ranks))
    mass = float(tot[hidx].sum() / s)
    auroc = float(roc_auc_score(h_mask, tot)); auprc = float(average_precision_score(h_mask, tot))
    ent = float(shannon(totn) / np.log(nh)); size_norm = float(mass / (len(hidx) / nh))
    true_c = h_xyz[hidx].mean(0); pred_c = (totn[:, None] * h_xyz).sum(0)
    cdist = float(np.linalg.norm(pred_c - true_c))
    exp_disp = float((totn * np.linalg.norm(h_xyz - true_c, axis=1)).sum())
    argmax_disp = float(np.linalg.norm(h_xyz[order[:, 0]] - true_c, axis=1).mean())
    # nearest among the 19 Beauchamp regions (descriptive identification; chance 1/19)
    rk = list(reg_cents); C = np.array([reg_cents[k] for k in rk])
    o19 = np.argsort(np.linalg.norm(C - pred_c, axis=1))
    nr19_hit = int(rk[o19[0]] == true_key); nr19_rank = int(np.where(np.array(rk)[o19] == true_key)[0][0]) + 1
    # PARCEL-SET permutation null (primary significance): does the true region capture
    # more routed mass than random human parcel sets of the same size?  fine-grained p.
    nreg = len(hidx)
    rand_mass = np.array([tot[rng.integers(0, nh, nreg)].sum() / s for _ in range(NPERM)])
    perm_p_mass = float((1 + np.sum(rand_mass >= mass)) / (NPERM + 1))
    # analytic chance / enrichment (top-k)
    p1 = len(hidx) / nh; ch1 = p1; ch5 = 1 - (1 - p1) ** 5; ch10 = 1 - (1 - p1) ** 10
    # rotation/spin null (secondary) on displacement
    R = Rotation.random(NSPIN, random_state=1).as_matrix()
    null_d = np.linalg.norm(pred_c[None, :] - (brain_c + (R @ (true_c - brain_c))), axis=1)
    spin_p = float((1 + np.sum(null_d <= cdist)) / (NSPIN + 1))
    # bootstrap (parcel-level) CIs
    bt1, btm, btd = [], [], []
    for _ in range(NBOOT):
        bi = rng.integers(0, nm, nm); bb = blk[bi]; bt = bb.sum(0); bs = bt.sum()
        bt1.append(np.isin(bb.argmax(1), hidx).mean()); btm.append(bt[hidx].sum() / bs)
        btd.append(np.linalg.norm(((bt / bs)[:, None] * h_xyz).sum(0) - true_c))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return dict(top1=top1, top5=top5, top10=top10, mean_rank=mean_rank, mass_in_region=mass,
                auroc=auroc, auprc=auprc, entropy=ent, size_norm=size_norm,
                centroid_disp_mm=cdist, expected_disp_mm=exp_disp, argmax_disp_mm=argmax_disp,
                nr19_hit=nr19_hit, nr19_rank=nr19_rank, nr19_n=len(rk),
                chance_top1=ch1, enrich_top1=float(top1 / max(ch1, 1e-9)), chance_top5=ch5, chance_top10=ch10,
                perm_p_mass=perm_p_mass, spin_p_disp=spin_p,
                n_mouse=int(nm), ci_top1=ci(bt1), ci_mass=ci(btm), ci_disp=ci(btd))

def build(M, H):
    name2lab = b.parse_dsurqe_tree(b.EXT / "AMBA/data/DSURQE_tree.json")
    pdsq = b.assign_dsurqe_labels(M, b.EXT / "AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc")
    hmem = b.assign_human_region_membership(H, b.HUMAN_REGION_MNI)
    h_xyz = H.var[["x", "y", "z"]].to_numpy(); brain_c = h_xyz.mean(0)
    pairs = {}
    for mn, hn in b.BEAUCHAMP_PAIRS:
        if mn not in name2lab: continue
        mm = np.isin(pdsq, list(name2lab[mn])); hm = hmem.get(hn)
        if hm is None or mm.sum() == 0 or hm.sum() == 0: continue
        pairs[f"{mn} -> {hn}"] = (mm, hm)
    reg_cents = {k: h_xyz[np.where(hm)[0]].mean(0) for k, (mm, hm) in pairs.items()}
    reg_masks = {k: hm for k, (mm, hm) in pairs.items()}
    return pairs, reg_cents, reg_masks, h_xyz, brain_c, pdsq

def score_all(pi, pairs, reg_cents, reg_masks, h_xyz, brain_c):
    out = {}
    for key, (mm, hm) in pairs.items():
        out[key] = battery(pi, mm, hm, h_xyz, reg_cents, reg_masks, key, brain_c)
    ks = list(out)
    for pkey, qkey in (("perm_p_mass", "perm_q_mass"), ("spin_p_disp", "spin_q_disp")):
        q = bh_fdr([out[k][pkey] for k in ks])
        for k, qq in zip(ks, q): out[k][qkey] = float(qq)
    w = np.array([out[k]["n_mouse"] for k in ks])
    agg = {m: float(np.average([out[k][m] for k in ks], weights=w)) for m in
           ("top1", "top5", "top10", "mass_in_region", "auroc", "auprc", "entropy",
            "centroid_disp_mm", "expected_disp_mm")}
    agg["nr19_hit_frac"] = float(np.mean([out[k]["nr19_hit"] for k in ks]))
    agg["perm_mass_sig_frac"] = float(np.mean([out[k]["perm_q_mass"] < 0.05 for k in ks]))
    agg["spin_disp_sig_frac"] = float(np.mean([out[k]["spin_q_disp"] < 0.05 for k in ks]))
    return {"per_region": out, "aggregate": agg}

def main():
    loro = "--loro" in sys.argv
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    pairs, reg_cents, reg_masks, h_xyz, brain_c, pdsq = build(M, H)

    if not loro:
        pi = load_pi().astype(np.float64)          # canonical coupling (pi_canonical.npy)
        res = score_all(pi, pairs, reg_cents, reg_masks, h_xyz, brain_c)
        for k, v in res["per_region"].items():
            print(f"  {k.split(' -> ')[0][:20]:20s} top1={v['top1']:.2f} auroc={v['auroc']:.2f} "
                  f"disp={v['centroid_disp_mm']:.0f} massP={v['perm_p_mass']:.3f} q={v['perm_q_mass']:.3f} "
                  f"spinq={v['spin_q_disp']:.3f} nr19={v['nr19_hit']}", flush=True)
        print("\nAGG:", {k: round(x, 3) for k, x in res["aggregate"].items()})
        (ROOT / "outputs/logs/beauchamp_metric_battery.json").write_text(json.dumps(res, indent=2, default=float))
        print("wrote beauchamp_metric_battery.json")
        return

    # ---- LORO mode (resumable) ----
    import time
    from otter.data.anchor_packs import build_default_pack_entries
    from otter.models import MultimodalFGW
    CACHE = ROOT / "outputs/logs/beauchamp_metric_battery_loro.json"
    GUARD = 34.0; t0 = time.time()
    entries = build_default_pack_entries(M.var, H.var, atlas_root=str(ROOT))
    g0m = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    g0h = H.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    apm = pd.to_numeric(M.var["anchor_pair_id"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    aph = pd.to_numeric(H.var["anchor_pair_id"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    emi = [np.asarray(e.mouse_indices, int) for e in entries]
    res = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [k for k in pairs if k not in res]
    print(f"LORO: {len(res)}/{len(pairs)} done; remaining {len(todo)}")
    for key in todo:
        if time.time() - t0 > GUARD:
            print("guard -> re-run to continue"); break
        mm, hm = pairs[key]
        pids = set(apm[g0m & mm & np.isfinite(apm)].astype(int).tolist()) or {-999}
        dm = g0m & np.isin(np.nan_to_num(apm, nan=-1).astype(int), list(pids))
        dh = g0h & np.isin(np.nan_to_num(aph, nan=-1).astype(int), list(pids))
        M.var["garin_anchor"] = g0m & ~dm; H.var["garin_anchor"] = g0h & ~dh
        ent_ho = [e for e, mi in zip(entries, emi) if not mm[mi].any()]
        m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7, epsilon=5e-3,
                          xyz_weight=0.5, lam_anchor=1.0, alpha=0.5)
        m.fit(M, H, Cm_SC=np.load(ROOT / "outputs/anndata/full_costs.npz")["Cm_SC"],
              Ch_SC=np.load(ROOT / "outputs/anndata/full_costs.npz")["Ch_SC"], region_anchors=ent_ho)
        M.var["garin_anchor"] = g0m; H.var["garin_anchor"] = g0h
        res[key] = battery(m.pi.astype(np.float64), mm, hm, h_xyz, reg_cents, reg_masks, key, brain_c)
        CACHE.write_text(json.dumps(res, indent=2, default=float))
        v = res[key]
        print(f"  {key.split(' -> ')[0][:20]:20s} top1={v['top1']:.2f} auroc={v['auroc']:.2f} "
              f"disp={v['centroid_disp_mm']:.0f} nr19={v['nr19_hit']}", flush=True)
    if not todo:
        # finalize FDR + aggregate
        ks = list(res)
        for pkey, qkey in (("perm_p_mass", "perm_q_mass"), ("spin_p_disp", "spin_q_disp")):
            q = bh_fdr([res[k][pkey] for k in ks])
            for k, qq in zip(ks, q): res[k][qkey] = float(qq)
        CACHE.write_text(json.dumps(res, indent=2, default=float))
        w = np.array([res[k]["n_mouse"] for k in ks])
        print("LORO AGG auroc=%.2f disp=%.0f top1=%.2f nr19=%.2f" % (
            np.average([res[k]["auroc"] for k in ks], weights=w),
            np.average([res[k]["centroid_disp_mm"] for k in ks], weights=w),
            np.average([res[k]["top1"] for k in ks], weights=w),
            np.mean([res[k]["nr19_hit"] for k in ks])))
        print("ALL DONE")

if __name__ == "__main__":
    main()
