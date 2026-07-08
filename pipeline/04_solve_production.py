"""Pipeline step 04, solve the production model and save π.

Uses the new sklearn-style API to fit MultimodalFGW (FC + SC, anchor +
xyz supervision) on full data and save the resulting coupling.

Outputs:
    outputs/coupling/pi_fc_plus_SC.npy, production π (1864 × 2094)
    outputs/coupling/pi_fc_plus_SC.json, config + fit info sidecar

Usage:
    python pipeline/04_solve_production.py
    python pipeline/04_solve_production.py --config fc_only       # baseline
    python pipeline/04_solve_production.py --config fc_plus_SC_selected
    python pipeline/04_solve_production.py --multistart            # 4 random + uniform inits
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached                          # noqa: E402
from homer.models import MultimodalFGW, SupervisedFGW       # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"; COUP.mkdir(parents=True, exist_ok=True)
LOGS = ROOT / "outputs" / "logs"
SELECTED_WEIGHTS = LOGS / "weight_selection_selected.json"


CONFIGS = {
    "fc_only":          dict(model_cls=SupervisedFGW,
                              kwargs=dict(epsilon=5e-3, xyz_weight=0.5)),
    "fc_plus_SC":       dict(model_cls=MultimodalFGW,
                              kwargs=dict(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                                          epsilon=5e-3, xyz_weight=0.5)),
    "fc_plus_SC_selected": dict(model_cls=MultimodalFGW, kwargs=None),
}


def load_selected_weights(path: Path) -> dict:
    """Load validation-selected production weights from 05i_weight_selection."""
    if not path.exists():
        raise FileNotFoundError(
            f"selected weights not found: {path}\n"
            "Run `python pipeline/05i_weight_selection.py` first, or pass "
            "`--selected-weights PATH`."
        )
    payload = json.loads(path.read_text())
    required = {"fc_weight", "sc_weight", "xyz_weight", "use_sc"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"selected weights file missing keys: {missing}")
    return {
        "use_sc": bool(payload["use_sc"]),
        "fc_weight": float(payload["fc_weight"]),
        "sc_weight": float(payload["sc_weight"]),
        "xyz_weight": float(payload["xyz_weight"]),
        "epsilon": 5e-3,
    }


def main(args):
    cfg = CONFIGS[args.config]
    cls = cfg["model_cls"]
    if args.config == "fc_plus_SC_selected":
        kwargs = load_selected_weights(Path(args.selected_weights))
    else:
        kwargs = dict(cfg["kwargs"])
    if args.multistart:
        kwargs["use_multistart"] = True
        kwargs["n_restarts"] = args.n_restarts

    print(f"Loading anndata from {ANN}")
    H, _ = load_cached("human", cache_dir=ANN)
    M, _ = load_cached("mouse", cache_dir=ANN)
    print(f"  mouse {M.uns['n_nodes']} nodes, human {H.uns['n_nodes']} nodes")

    fit_kwargs = {}
    if cls is MultimodalFGW and kwargs.get("use_sc", False):
        d = np.load(ANN / "full_costs.npz")
        fit_kwargs["Cm_SC"] = d["Cm_SC"]
        fit_kwargs["Ch_SC"] = d["Ch_SC"]
        print("  loaded SC cost matrices from full_costs.npz")

    print(f"Fitting {cls.__name__}({kwargs})...")
    model = cls(**kwargs)
    t0 = time.time()
    model.fit(M, H, **fit_kwargs)
    print(f"  done in {time.time() - t0:.1f}s, {model!r}")

    out_path = COUP / f"pi_{args.config}.npy"
    model.save(out_path)
    print(f"\nsaved → {out_path}")
    print(f"  sidecar → {out_path.with_suffix('.json')}")

    # Quick metrics summary
    print("\nFull anchor recovery (sanity check, no holdouts):")
    res = model.evaluate(eval_kind="anchor")
    print(f"  top1={res['top1']:.0%}  top5={res['top5']:.0%}  pair_id={res['pair_id']:.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fc_plus_SC", choices=list(CONFIGS.keys()))
    ap.add_argument("--selected-weights", default=str(SELECTED_WEIGHTS),
                    help="JSON produced by pipeline/05i_weight_selection.py")
    ap.add_argument("--multistart", action="store_true")
    ap.add_argument("--n-restarts", type=int, default=4)
    main(ap.parse_args())
