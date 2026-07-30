#!/usr/bin/env python3
"""CANONICAL optimized coupling: warped spatial term + region packs + supervision,
with (epsilon, xyz_weight) selected by held-out (nested CV) external Beauchamp recovery.

For every grid cell we also record how the hyperparameter choice affects
reconstruction-coverage biology: region-level Spearman(recon_cov, Xu2020 expansion) and
the ContB(dlPFC) deficit in SD.

Modes (chunked so no bash call exceeds the timeout):
  python 23_canonical_pi.py --fit  [--chunk N]   # fit next N undone grid cells, append to JSON
  python 23_canonical_pi.py --select             # nested-CV + all-data pick + canonical refit

Grid: xyz_weight in {0.1,0.25,0.5,0.75,1.0} x epsilon in {0.005,0.02,0.05,0.1,0.2}  (25 cells).
Everything is written to outputs/logs/section5_canonical_sweep.json.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/section5_coverage_rigor"))

from homer.data import load_cached                                   # noqa: E402
from homer.data.anchors import get_anchor_index                     # noqa: E402
from homer.data.anchor_packs import build_default_pack_entries      # noqa: E402
from homer.models import MultimodalFGW                              # noqa: E402
from beauchamp_scorer import BeauchampScorer                        # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
LOG  = ROOT / "outputs" / "logs"
COUP = ROOT / "outputs" / "coupling"
SWEEP_JSON = LOG / "section5_canonical_sweep.json"
TMP = Path("/var/tmp")

XYZ_WEIGHTS = [0.1, 0.25, 0.5, 0.75, 1.0]
# Extended 2026-07-21 from [0.005, 0.05]. The two-point grid could not support the
# claim that epsilon is chosen robustly, and a separate five-point sweep scored on a
# different criterion (L/R reliability) had picked 0.2. One grid, one criterion.
EPSILONS    = [0.005, 0.02, 0.05, 0.1, 0.2]
GRID = [(w, e) for w in XYZ_WEIGHTS for e in EPSILONS]   # 25 cells


def cell_key(w, e):
    return f"w{w}_e{e}"


# --------------------------------------------------------------------------- setup
def build_context():
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")

    # ---- WARPED spatial term (full 42-anchor thin-plate-spline) ----
    im = get_anchor_index(M.var); ih = get_anchor_index(H.var)
    hlut = {(int(p), str(h)): int(k) for k, p, h in zip(ih.pos, ih.pair_ids, ih.hemispheres)}
    trip = [(int(mp), hlut[(int(pid), str(hemi))])
            for mp, pid, hemi in zip(im.pos, im.pair_ids, im.hemispheres)
            if (int(pid), str(hemi)) in hlut]
    mxyz = M.var[["x", "y", "z"]].to_numpy(float)
    hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    src = mxyz[[a for a, b in trip]]; dst = hxyz[[b for a, b in trip]]
    warp = RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=1e-3)
    wm = warp(mxyz)
    d = np.sqrt(((wm[:, None, :] - hxyz[None, :, :]) ** 2).sum(-1))
    M_xyz_warp = (d / max(d.max(), 1e-9)).astype(np.float64)
    print(f"[warp] {len(trip)} anchor pairs -> M_xyz_warp {M_xyz_warp.shape}")

    entries = build_default_pack_entries(M.var, H.var, atlas_root=ROOT)
    print(f"[packs] {len(entries)} region-anchor entries")

    sc = BeauchampScorer()
    print(f"[beauchamp] {len(sc.pairs)} scorable pairs")

    # ---- reconstruction-coverage biology helpers ----
    Mfc = np.asarray(M.uns["fc_mean"], float); Hfc = np.asarray(H.uns["fc_mean"], float)
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    rows = [l.split("\t") for l in (ANN / "_schaefer_order.txt").read_text().splitlines() if l.strip()]
    net = np.array([{int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}.get(int(k), "?") for k in nr])
    mye = np.asarray(json.loads((LOG / "buckner_krienen_2013_tethering.json").read_text())["myelin_per_parcel"], float)
    b = json.loads((LOG / "section5_evolution_battery.json").read_text())
    xu = dict(zip(np.asarray(b["Xu2020 mouse→human expansion"]["schaefer_ids"], int),
                  np.asarray(b["Xu2020 mouse→human expansion"]["map_values"], float)))

    return dict(M=M, H=H, costs=costs, M_xyz_warp=M_xyz_warp, entries=entries, sc=sc,
                Mfc=Mfc, Hfc=Hfc, nr=nr, net=net, mye=mye, xu=xu)


def recon_cov(pi, Mfc, Hfc):
    ph = pi.sum(0); pit = pi / np.maximum(ph, 1e-300)
    pred = pit.T @ Mfc @ pit
    n = pred.shape[0]; out = np.full(n, np.nan)
    for j in range(n):
        a = pred[j].copy(); b = Hfc[j].copy(); a[j] = np.nan; b[j] = np.nan
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10 and a[ok].std() > 1e-9:
            out[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


def biology(cov, nr, net, mye, xu):
    # region-level Spearman(recon_cov, Xu2020 expansion)
    eids = [k for k in range(1, 401) if (nr == k).any() and k in xu]
    cc = np.array([np.nanmean(cov[nr == k]) for k in eids])
    ev = np.array([xu[k] for k in eids])
    r_exp = spearmanr(cc, ev, nan_policy="omit").statistic
    # ContB (dlPFC) deficit in SD over cortical parcels (finite myelin)
    m = np.isfinite(cov) & np.isfinite(mye)
    z = (cov[m] - np.nanmean(cov[m])) / np.nanstd(cov[m])
    sel = net[m] == "ContB"
    contb = float(np.nanmean(z[sel]) - np.nanmean(z[~sel]))
    return float(r_exp), contb


def fit_cell(ctx, w, e):
    t0 = time.time()
    model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                          epsilon=e, xyz_weight=w, lam_anchor=1.0)
    model.fit(ctx["M"], ctx["H"], Cm_SC=ctx["costs"]["Cm_SC"], Ch_SC=ctx["costs"]["Ch_SC"],
              M_xyz=ctx["M_xyz_warp"], region_anchors=ctx["entries"])
    pi = model.pi.astype(np.float64)
    np.save(TMP / f"pi_canon_{cell_key(w, e)}.npy", pi)

    res = ctx["sc"].score(pi)
    agg = res["__aggregate__"]
    per_pair = {p: float(res[p]["top1"]) for p in ctx["sc"].pairs if p in res}

    cov = recon_cov(pi, ctx["Mfc"], ctx["Hfc"])
    r_exp, contb = biology(cov, ctx["nr"], ctx["net"], ctx["mye"], ctx["xu"])

    return {
        "xyz_weight": w, "epsilon": e,
        "beauchamp_top1": float(agg["top1"]), "beauchamp_top5": float(agg["top5"]),
        "beauchamp_top10": float(agg["top10"]),
        "n_pairs": int(agg["n_pairs"]),
        "per_pair_top1": per_pair,
        "expansion_rho": r_exp, "ContB_deficit_SD": contb,
        "loss": float(model.fit_info_.loss), "converged": bool(model.fit_info_.converged),
        "pi_file": str(TMP / f"pi_canon_{cell_key(w, e)}.npy"),
        "fit_seconds": round(time.time() - t0, 1),
    }


def load_state():
    if SWEEP_JSON.exists():
        return json.loads(SWEEP_JSON.read_text())
    return {"_def": "canonical (epsilon,xyz_weight) sweep: warp+packs+supervision; "
                    "Beauchamp held-out CV selection + recon-coverage biology per cell",
            "grid": {"xyz_weight": XYZ_WEIGHTS, "epsilon": EPSILONS}, "cells": {}}
    # NB: `grid` is written once at init. If the grid constants change later, the
    # recorded grid will disagree with the fitted cells until it is refreshed.
    state["grid"] = {"xyz_weight": XYZ_WEIGHTS, "epsilon": EPSILONS}


def save_state(state):
    SWEEP_JSON.write_text(json.dumps(state, indent=2))


# --------------------------------------------------------------------------- fit
def run_fit(chunk):
    state = load_state()
    todo = [(w, e) for (w, e) in GRID if cell_key(w, e) not in state["cells"]]
    if not todo:
        print(f"all {len(GRID)} cells already fitted."); return
    ctx = build_context()
    for (w, e) in todo[:chunk]:
        print(f"\n=== fitting cell {cell_key(w, e)} ===")
        r = fit_cell(ctx, w, e)
        state["cells"][cell_key(w, e)] = r
        save_state(state)
        print(f"  top1={r['beauchamp_top1']:.3f} top5={r['beauchamp_top5']:.3f} "
              f"exp_rho={r['expansion_rho']:+.3f} ContB={r['ContB_deficit_SD']:+.2f}SD "
              f"({r['fit_seconds']}s)")
    remaining = [c for (w, e) in GRID if (c := cell_key(w, e)) not in state["cells"]]
    print(f"\ndone this call. remaining cells: {remaining}")


# --------------------------------------------------------------------------- select
def run_select():
    state = load_state()
    cells = state["cells"]
    if len(cells) < len(GRID):
        print(f"only {len(cells)}/{len(GRID)} cells fitted; run --fit first."); return

    # union of pairs (all cells share the same scorable set)
    all_pairs = sorted(set().union(*[set(cells[c]["per_pair_top1"]) for c in cells]))
    cell_ids = [cell_key(w, e) for (w, e) in GRID]

    # matrix cells x pairs of per-pair top1
    P = np.array([[cells[c]["per_pair_top1"].get(p, np.nan) for p in all_pairs] for c in cell_ids])
    print(f"[cv] {len(all_pairs)} pairs, {len(cell_ids)} cells")

    # 5-fold nested CV, seed 0
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(all_pairs))
    folds = np.array_split(idx, 5)
    heldout_top1, selected = [], []
    for f, test in enumerate(folds):
        train = np.setdiff1d(idx, test)
        train_mean = np.nanmean(P[:, train], axis=1)   # per-cell mean top1 on train pairs
        best = int(np.argmax(train_mean))
        ho = float(np.nanmean(P[best, test]))
        heldout_top1.append(ho); selected.append(cell_ids[best])
        print(f"  fold {f}: select {cell_ids[best]} (train {train_mean[best]:.3f}) "
              f"-> held-out top1 {ho:.3f}")
    mean_heldout = float(np.mean(heldout_top1))
    vals, counts = np.unique(selected, return_counts=True)
    modal = str(vals[int(np.argmax(counts))])

    # all-data pick = cell maximising mean per-pair top1 across ALL pairs -> deploy this
    all_mean = np.nanmean(P, axis=1)
    deploy_id = cell_ids[int(np.argmax(all_mean))]
    dw, de = next((w, e) for (w, e) in GRID if cell_key(w, e) == deploy_id)
    print(f"\n[cv] mean held-out top1 = {mean_heldout:.3f}; modal cell = {modal}")
    print(f"[deploy] all-data best cell = {deploy_id} (mean per-pair top1 {all_mean.max():.3f})")

    # ---- canonical refit at deploy cell, save to coupling/pi_canonical.npy ----
    ctx = build_context()
    print(f"\n[canonical] refitting at xyz_weight={dw}, epsilon={de} ...")
    r = fit_cell(ctx, dw, de)
    pi = np.load(r["pi_file"])
    COUP.mkdir(exist_ok=True)
    np.save(COUP / "pi_canonical.npy", pi)
    print(f"[canonical] saved {COUP / 'pi_canonical.npy'}  shape={pi.shape}")

    state["nested_cv"] = {
        "n_folds": 5, "seed": 0, "n_pairs": len(all_pairs),
        "per_fold_selected": selected, "per_fold_heldout_top1": heldout_top1,
        "mean_heldout_top1": mean_heldout, "modal_selected_cell": modal,
    }
    state["deploy"] = {
        "cell": deploy_id, "xyz_weight": dw, "epsilon": de,
        "beauchamp_top1": r["beauchamp_top1"], "beauchamp_top5": r["beauchamp_top5"],
        "beauchamp_top10": r["beauchamp_top10"],
        "heldout_cv_top1": mean_heldout,
        "expansion_rho": r["expansion_rho"], "ContB_deficit_SD": r["ContB_deficit_SD"],
        "loss": r["loss"], "converged": r["converged"],
        "pi_path": str(COUP / "pi_canonical.npy"),
    }
    save_state(state)
    print(f"\n[done] canonical: top1={r['beauchamp_top1']:.3f} top5={r['beauchamp_top5']:.3f} "
          f"heldout_cv={mean_heldout:.3f} exp_rho={r['expansion_rho']:+.3f} "
          f"ContB={r['ContB_deficit_SD']:+.2f}SD")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--chunk", type=int, default=3)
    a = ap.parse_args()
    if a.fit:
        run_fit(a.chunk)
    elif a.select:
        run_select()
    else:
        ap.print_help()
