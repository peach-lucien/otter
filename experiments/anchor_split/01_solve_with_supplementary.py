"""SPLIT-1: re-solve production π with one supplementary anchor (narrow M1).

Loads the existing AnnDatas, applies the M1-narrow supplementary anchor (pid=22)
from `config/supplementary_anchors_motor.yaml`, re-fits MultimodalFGW with
exactly the same hyperparameters as production, then re-runs the Beauchamp
external-validation comparison.

Output:
    outputs/coupling/pi_fc_plus_SC_with_M1.npy   (+ sidecar JSON)
    outputs/logs/beauchamp_validation_with_M1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import anndata as ad

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached                                       # noqa: E402
from homer.data.supplementary_anchors import (                           # noqa: E402
    parse_supplementary_anchors_config,
    apply_supplementary_anchors,
    summarize_supplementary_anchors,
)
from homer.data.anchors import get_anchor_index                          # noqa: E402
from homer.models import MultimodalFGW                                   # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG  = ROOT / "outputs" / "logs"


def make_augmented_anndata(species: str, M_or_H, var_aug):
    """Wrap (X, var, uns) into a fresh AnnData with the modified var table."""
    new = ad.AnnData(
        X=M_or_H.X if M_or_H.X is not None else None,
        var=var_aug,
        obs=M_or_H.obs.copy() if hasattr(M_or_H, "obs") else None,
        uns={k: v for k, v in M_or_H.uns.items()},
    )
    return new


def main(args):
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)

    print(f"Loading supplementary anchors from {args.config}")
    entries = parse_supplementary_anchors_config(args.config, M.var, H.var)
    print(summarize_supplementary_anchors(entries, M.var, H.var))

    var_m_aug, var_h_aug = apply_supplementary_anchors(M.var, H.var, entries)
    M_aug = make_augmented_anndata("mouse", M, var_m_aug)
    H_aug = make_augmented_anndata("human", H, var_h_aug)

    idx_m = get_anchor_index(M_aug.var); idx_h = get_anchor_index(H_aug.var)
    print(f"\nAnchors after supplementary apply: {len(idx_m)} mouse, {len(idx_h)} human")
    print(f"Pair ids: {sorted(set(idx_m.pair_ids))}")
    assert idx_m.keys == idx_h.keys, "anchor key orderings differ"

    # Solve with same hyperparams as production (pi_fc_plus_SC.npy)
    costs = np.load(ANN / "full_costs.npz")
    print(f"\nSolving MultimodalFGW with augmented anchors ({len(idx_m)} pairs)...")
    t = time.time()
    model = MultimodalFGW(
        use_sc=True, sc_weight=0.3, fc_weight=0.7,
        epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0,
    )
    model.fit(M_aug, H_aug, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"])
    elapsed = time.time() - t
    print(f"  Solved in {elapsed:.1f}s, loss={model.fit_info_.loss:.6f}")

    # Save π + sidecar
    out_pi = COUP / args.out_pi
    np.save(out_pi, model.pi.astype(np.float32))
    sidecar = {
        "model_class": "MultimodalFGW",
        "config": dict(model.config),
        "supplementary_anchors": [
            {"pair_id": e.pair_id, "label": e.label,
             "L_mouse": str(M.var.index[e.L_mouse_idx]),
             "L_human": str(H.var.index[e.L_human_idx]),
             "R_mouse": str(M.var.index[e.R_mouse_idx]),
             "R_human": str(H.var.index[e.R_human_idx])}
            for e in entries
        ],
        "fit_info": {
            "loss": float(model.fit_info_.loss),
            "n_iter": model.fit_info_.n_iter,
            "converged": model.fit_info_.converged,
            "extra": dict(model.fit_info_.extra),
        },
        "shape": list(model.pi.shape),
        "pi_file": args.out_pi,
        "elapsed_s": round(elapsed, 1),
    }
    (out_pi.with_suffix(".json")).write_text(
        json.dumps(sidecar, indent=2, default=str))
    print(f"\nSaved {out_pi}  (+ sidecar)")

    # Quick anchor recovery check
    from homer.data.anchors import metrics_summary
    pi_anchor_block = model.pi[np.ix_(idx_m.pos, idx_h.pos)]
    print(f"\nAnchor block top-1 (incl. supplementary): "
          f"{metrics_summary(pi_anchor_block, idx_m, idx_h)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/supplementary_anchors_motor.yaml")
    ap.add_argument("--out-pi", default="pi_fc_plus_SC_with_M1.npy",
                    help="filename in outputs/coupling/")
    main(ap.parse_args())
