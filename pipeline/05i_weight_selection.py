"""Select FC/SC/xyz weights by validation instead of hardcoding production.

This script sweeps a small, explicit hyperparameter grid:

    fc_sc_ratio = fc_weight / (fc_weight + sc_weight)
    xyz_weight

For each candidate it runs leave-one-network-out anchor CV. By default it also
fits the full model and evaluates brain-wide FC translation, so the selected
configuration is not chosen from anchor recovery alone.

Outputs:
    outputs/logs/weight_selection.json
    outputs/logs/weight_selection.csv
    outputs/logs/weight_selection_selected.json

Usage:
    python pipeline/05i_weight_selection.py
    python pipeline/05i_weight_selection.py --networks visual,brainstem
    python pipeline/05i_weight_selection.py --no-translation
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homer.data import load_cached  # noqa: E402
from homer.eval.anchor_cv import anchor_loo_cv  # noqa: E402
from homer.models import MultimodalFGW  # noqa: E402


ANN = ROOT / "outputs" / "anndata"
LOG = ROOT / "outputs" / "logs"
LOG.mkdir(parents=True, exist_ok=True)

DEFAULT_FC_RATIOS = "1.0,0.9,0.8,0.7,0.6,0.5,0.3,0.0"
DEFAULT_XYZ_WEIGHTS = "0.0,0.25,0.5,0.75,1.0"


def _parse_float_grid(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("grid must contain at least one value")
    return vals


def _finite(x, default: float = 0.0) -> float:
    try:
        x = float(x)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _score(row: dict, weights: dict[str, float]) -> float:
    """Default validation score.

    All inputs are already on roughly [0, 1] scales:
      - anchor_top1: hit rate
      - fc_translation_r: Pearson r
      - fc_within_r: Pearson r within human networks
    """
    return (
        weights["anchor_top1"] * _finite(row.get("anchor_top1"))
        + weights["fc_translation_r"] * _finite(row.get("fc_translation_r"))
        + weights["fc_within_r"] * _finite(row.get("fc_within_r"))
    )


def _candidate_key(fc_ratio: float, xyz_weight: float) -> str:
    return f"fc{fc_ratio:.3f}_sc{1.0 - fc_ratio:.3f}_xyz{xyz_weight:.3f}"


def _model_kwargs(fc_ratio: float, xyz_weight: float, args) -> dict:
    sc_weight = max(0.0, 1.0 - fc_ratio)
    return {
        "use_sc": sc_weight > 0,
        "fc_weight": max(0.0, fc_ratio),
        "sc_weight": sc_weight,
        "xyz_weight": xyz_weight,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "lam_anchor": args.lam_anchor,
        "use_multistart": args.n_restarts > 0,
        "n_restarts": args.n_restarts,
    }


def _translation_eval(mouse_ad, human_ad, fit_kwargs: dict, model_kwargs: dict) -> dict:
    model = MultimodalFGW(**model_kwargs)
    model.fit(mouse_ad, human_ad, **fit_kwargs)
    return model.evaluate(eval_kind="translation")


def main(args) -> int:
    fc_ratios = _parse_float_grid(args.fc_ratios)
    xyz_weights = _parse_float_grid(args.xyz_weights)
    networks = args.networks.split(",") if args.networks else None
    score_weights = {
        "anchor_top1": args.score_anchor,
        "fc_translation_r": args.score_fc,
        "fc_within_r": args.score_within,
    }

    human_ad, _ = load_cached("human", cache_dir=ANN)
    mouse_ad, _ = load_cached("mouse", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")
    fit_kwargs_sc = {"Cm_SC": costs["Cm_SC"], "Ch_SC": costs["Ch_SC"]}

    result_path = LOG / "weight_selection.json"
    csv_path = LOG / "weight_selection.csv"
    selected_path = LOG / "weight_selection_selected.json"

    state = json.loads(result_path.read_text()) if result_path.exists() and not args.recompute else {
        "experiment": "weight_selection",
        "search_space": {
            "fc_ratios": fc_ratios,
            "xyz_weights": xyz_weights,
            "networks": networks or "all",
        },
        "score_weights": score_weights,
        "candidates": {},
    }

    for fc_ratio in fc_ratios:
        if not 0.0 <= fc_ratio <= 1.0:
            raise ValueError(f"fc ratio must be in [0, 1], got {fc_ratio}")
        for xyz_weight in xyz_weights:
            key = _candidate_key(fc_ratio, xyz_weight)
            if key in state["candidates"] and not args.recompute:
                print(f"{key}: cached")
                continue

            model_kwargs = _model_kwargs(fc_ratio, xyz_weight, args)
            fit_kwargs = fit_kwargs_sc if model_kwargs["use_sc"] else {}
            print(f"\n=== {key} ===")
            print(f"  model kwargs: {model_kwargs}")
            t0 = time.time()
            cv = anchor_loo_cv(
                lambda kw=model_kwargs: MultimodalFGW(**kw),
                mouse_ad,
                human_ad,
                networks=networks,
                fit_kwargs=fit_kwargs,
                verbose=True,
            )
            weighted = cv["weighted"]

            translation = {}
            if not args.no_translation:
                translation = _translation_eval(mouse_ad, human_ad, fit_kwargs, model_kwargs)

            row = {
                "key": key,
                "fc_ratio": float(fc_ratio),
                "fc_weight": float(model_kwargs["fc_weight"]),
                "sc_weight": float(model_kwargs["sc_weight"]),
                "xyz_weight": float(xyz_weight),
                "anchor_top1": weighted.get("top1"),
                "anchor_top5": weighted.get("top5"),
                "anchor_pair_id": weighted.get("pair_id"),
                "anchor_mean_rank": weighted.get("mean_rank"),
                "anchor_mean_xyz_dist": weighted.get("mean_xyz_dist"),
                "fc_translation_r": translation.get("pearson_r_overall"),
                "fc_within_r": translation.get("pearson_r_within_net"),
                "fc_cross_r": translation.get("pearson_r_cross_net"),
                "n_human_nodes_kept": translation.get("n_human_nodes_kept"),
                "elapsed_s": round(time.time() - t0, 1),
            }
            row["score"] = _score(row, score_weights)
            state["candidates"][key] = {
                "summary": row,
                "anchor_cv": cv,
                "translation": translation,
                "model_kwargs": model_kwargs,
            }
            result_path.write_text(json.dumps(state, indent=2, default=float))
            print(
                f"  score={row['score']:.4f} "
                f"top1={_finite(row['anchor_top1']):.1%} "
                f"fc_r={_finite(row['fc_translation_r']):.3f} "
                f"within={_finite(row['fc_within_r']):.3f}"
            )

    rows = [c["summary"] for c in state["candidates"].values()]
    rows.sort(key=lambda r: (r["score"], r["anchor_top1"], -_finite(r["anchor_mean_rank"], 99)), reverse=True)
    state["ranked_keys"] = [r["key"] for r in rows]
    state["selected"] = rows[0] if rows else None
    result_path.write_text(json.dumps(state, indent=2, default=float))

    if rows:
        fields = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        selected = rows[0]
        selected_payload = {
            "source": str(result_path.relative_to(ROOT)),
            "selection_key": selected["key"],
            "selection_score": selected["score"],
            "score_weights": score_weights,
            "fc_weight": selected["fc_weight"],
            "sc_weight": selected["sc_weight"],
            "xyz_weight": selected["xyz_weight"],
            "use_sc": selected["sc_weight"] > 0,
            "metrics": selected,
        }
        selected_path.write_text(json.dumps(selected_payload, indent=2, default=float))
        print(f"\nselected: {selected['key']} score={selected['score']:.4f}")
        print(f"wrote {result_path}")
        print(f"wrote {csv_path}")
        print(f"wrote {selected_path}")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc-ratios", default=DEFAULT_FC_RATIOS,
                    help="comma-separated FC ratios; SC ratio is 1-FC")
    ap.add_argument("--xyz-weights", default=DEFAULT_XYZ_WEIGHTS,
                    help="comma-separated xyz weights")
    ap.add_argument("--networks", default=None,
                    help="optional comma-separated LONO network subset")
    ap.add_argument("--no-translation", action="store_true",
                    help="skip full-model brain-wide FC translation metric")
    ap.add_argument("--recompute", action="store_true",
                    help="ignore cached candidates")
    ap.add_argument("--score-anchor", type=float, default=0.45)
    ap.add_argument("--score-fc", type=float, default=0.35)
    ap.add_argument("--score-within", type=float, default=0.20)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--epsilon", type=float, default=5e-3)
    ap.add_argument("--lam-anchor", type=float, default=1.0)
    ap.add_argument("--n-restarts", type=int, default=0,
                    help="if >0, enable multistart with this many random starts")
    raise SystemExit(main(ap.parse_args()))
