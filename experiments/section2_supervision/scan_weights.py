#!/usr/bin/env python3
"""Full grid scan of FGW cost weights (alpha, xyz_weight, FC/SC balance).

5 values per weight = 125 combos. Per combo, two fits:
  full   : all anchors (Garin + packs)          -> BEAUCHAMP accuracy (understanding)
  genOOB : ALL benchmark-region curation removed -> GENERALISATION objective
           (circularity-free: the 19 regions are scored with their own anchors gone).
Aggregates parcel-weighted over the 19 Beauchamp pairs.

Run (from the repo's homer/ dir, with the HOMER scientific env active):
    cd homer
    PYTHONPATH=src python ../manuscript/figures/fig_2_ED/scan_weights.py

Resumable: caches to outputs/logs/scan_weights.json; re-run to continue / finish.
Prints a summary table (best generalisation, best Beauchamp) at the end.
"""
import sys, json, time, importlib.util, itertools
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # repo/homer
CACHE = ROOT / "outputs" / "logs" / "scan_weights.json"
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached
from homer.data.anchor_packs import build_default_pack_entries
from homer.models import MultimodalFGW
spec = importlib.util.spec_from_file_location("b05f", ROOT / "pipeline/05f_beauchamp_validation.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

ALPHA = [0.1, 0.3, 0.5, 0.7, 0.9]     # connectivity(GW) vs feature(W); 0.9 = connectivity-heavy
XYZ   = [0.0, 0.25, 0.5, 0.75, 1.0]   # spatial term weight in the feature cost
FCFR  = [0.1, 0.3, 0.5, 0.7, 0.9]     # FC fraction of the relational cost; SC = 1 - FC

def metrics(pi, m_mask, h_mask, h_xyz):
    hidx = np.where(h_mask)[0]; true_c = h_xyz[hidx].mean(0); hset = set(hidx.tolist())
    blk = pi[m_mask]; tot = blk.sum(0); s = tot.sum()
    if s <= 0: return None
    am = blk.argmax(1); tk = np.argsort(-blk, 1)[:, :10]
    top1 = np.isin(am, hidx).mean()
    top5 = np.mean([any(t in hset for t in tk[i, :5]) for i in range(len(am))])
    top10 = np.mean([any(t in hset for t in tk[i]) for i in range(len(am))])
    pred_c = ((tot / s)[:, None] * h_xyz).sum(0)
    return dict(top1=float(top1), top5=float(top5), top10=float(top10),
                mass=float(tot[h_mask].sum() / s),
                cdist=float(np.linalg.norm(pred_c - true_c)), n=int(m_mask.sum()))

def agg(rows):
    w = np.array([r["n"] for r in rows])
    return {k: float(np.average([r[k] for r in rows], weights=w))
            for k in ("top1", "top5", "top10", "mass", "cdist")}

def fit(M, H, costs, entries, a, x, f):
    m = MultimodalFGW(use_sc=True, fc_weight=f, sc_weight=1 - f, epsilon=5e-3,
                      lam_anchor=1.0, alpha=a, xyz_weight=x)
    m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"], region_anchors=entries)
    return m.pi

def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    costs = np.load(ROOT / "outputs/anndata/full_costs.npz")
    entries = build_default_pack_entries(M.var, H.var, atlas_root=str(ROOT))
    name2lab = b.parse_dsurqe_tree(b.EXT / "AMBA/data/DSURQE_tree.json")
    pdsq = b.assign_dsurqe_labels(M, b.EXT / "AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc")
    hmem = b.assign_human_region_membership(H, b.HUMAN_REGION_MNI)
    h_xyz = H.var[["x", "y", "z"]].to_numpy()
    pairs = {}
    for mn, hn in b.BEAUCHAMP_PAIRS:
        if mn not in name2lab: continue
        mm = np.isin(pdsq, list(name2lab[mn])); hm = hmem.get(hn)
        if hm is None or mm.sum() == 0 or hm.sum() == 0: continue
        pairs[f"{mn}->{hn}"] = (mm, hm)

    any_bench = np.zeros(len(M.var), bool)
    for mm, _ in pairs.values(): any_bench |= mm
    g0m = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    g0h = H.var["garin_anchor"].fillna(False).to_numpy().astype(bool)
    apm = pd.to_numeric(M.var["anchor_pair_id"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    aph = pd.to_numeric(H.var["anchor_pair_id"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    fin = np.isfinite(apm)
    bench_pids = set(apm[g0m & any_bench & fin].astype(int).tolist()) or {-999}
    dm = g0m & np.isin(np.nan_to_num(apm, nan=-1).astype(int), list(bench_pids))
    dh = g0h & np.isin(np.nan_to_num(aph, nan=-1).astype(int), list(bench_pids))
    emi = [np.asarray(e.mouse_indices, int) for e in entries]
    ent_oob = [e for e, mi in zip(entries, emi) if not any_bench[mi].any()]

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    combos = list(itertools.product(ALPHA, XYZ, FCFR))
    todo = [c for c in combos if f"{c[0]}_{c[1]}_{c[2]}" not in res]
    print(f"{len(res)}/{len(combos)} done; running {len(todo)} ...", flush=True)
    for i, (a, x, f) in enumerate(todo):
        t = time.time()
        pif = fit(M, H, costs, entries, a, x, f)
        bea = agg([metrics(pif, mm, hm, h_xyz) for mm, hm in pairs.values()])
        M.var["garin_anchor"] = g0m & ~dm; H.var["garin_anchor"] = g0h & ~dh
        pig = fit(M, H, costs, ent_oob, a, x, f)
        M.var["garin_anchor"] = g0m; H.var["garin_anchor"] = g0h
        gen = agg([metrics(pig, mm, hm, h_xyz) for mm, hm in pairs.values()])
        res[f"{a}_{x}_{f}"] = dict(alpha=a, xyz=x, fcfr=f, bea=bea, gen=gen)
        CACHE.write_text(json.dumps(res, indent=2, default=float))
        print(f"  [{len(res)}/{len(combos)}] a={a} xyz={x} fc={f} | "
              f"BEA top1={bea['top1']:.2f} disp={bea['cdist']:.0f} | "
              f"GEN disp={gen['cdist']:.0f} mass={gen['mass']:.2f}  ({time.time()-t:.0f}s)", flush=True)

    rows = [(v['alpha'], v['xyz'], v['fcfr'], v['bea'], v['gen']) for v in res.values()]
    print(f"\n=== ALL {len(rows)} combos done. Written to {CACHE} ===")
    print("\nBest GENERALISATION (lowest gen disp):")
    for a, x, f, be, ge in sorted(rows, key=lambda z: z[4]['cdist'])[:8]:
        print(f"  a={a} xyz={x} fc={f} | GEN disp={ge['cdist']:.0f} mass={ge['mass']:.2f} | "
              f"BEA top1={be['top1']:.2f} disp={be['cdist']:.0f}")
    print("\nBest BEAUCHAMP (top1):")
    for a, x, f, be, ge in sorted(rows, key=lambda z: -z[3]['top1'])[:8]:
        print(f"  a={a} xyz={x} fc={f} | BEA top1={be['top1']:.2f} disp={be['cdist']:.0f} | "
              f"GEN disp={ge['cdist']:.0f}")

if __name__ == "__main__":
    main()
