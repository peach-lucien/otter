"""Benchmark anchor-only interpolation baselines against HOMER logs.

This script creates probabilistic couplings using only visible Garin anchors,
then evaluates them with the same leave-one-network-out held-anchor metrics used
by HOMER's FGW comparisons. It deliberately avoids FC/SC in the baseline
construction.

Outputs are written next to this file under ``results/``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from homer.data import DataNotFound, load_cached  # noqa: E402
from homer.data.anchors import get_anchor_index, held_out_metrics_graded  # noqa: E402
from homer.data.networks import NETWORKS, PAIRID_TO_NETWORK, assign_networks  # noqa: E402
from homer.eval.full_space_metrics import full_space_metrics  # noqa: E402
from homer.eval.translation import fc_translation_quality  # noqa: E402


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ANN = ROOT / "outputs" / "anndata"
LOGS = ROOT / "outputs" / "logs"


def _xyz(var) -> np.ndarray:
    return var[["x", "y", "z"]].to_numpy(dtype=np.float64)


def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = (a * a).sum(axis=1, keepdims=True)
    bb = (b * b).sum(axis=1, keepdims=True).T
    d2 = aa + bb - 2.0 * (a @ b.T)
    return np.sqrt(np.clip(d2, 0.0, None))


def _auto_tau(anchor_xyz: np.ndarray, *, fallback: float = 1.0) -> float:
    """Median nearest-neighbour anchor distance, robust default for kernels."""
    if len(anchor_xyz) < 2:
        return fallback
    d = _pairwise_dist(anchor_xyz, anchor_xyz)
    np.fill_diagonal(d, np.inf)
    nn = np.min(d, axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return fallback
    return float(max(np.median(nn), 1e-6))


def _normalise_rows(a: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    return a / a.sum(axis=1, keepdims=True).clip(min=eps)


def _row_scaled_coupling(row_prob: np.ndarray) -> np.ndarray:
    """Convert row-probabilities to HOMER-style source-marginal coupling."""
    n_m = row_prob.shape[0]
    return row_prob / float(n_m)


def _visible_anchor_indices(idx_m, visible_pair_ids: Iterable[int]) -> np.ndarray:
    visible = set(int(p) for p in visible_pair_ids)
    return np.array([k for k, pid in enumerate(idx_m.pair_ids) if int(pid) in visible],
                    dtype=np.int64)


def build_anchor_baseline(
    model: str,
    mouse_ad,
    human_ad,
    *,
    visible_pair_ids: Iterable[int],
    tau_mouse: float | None = None,
    tau_human: float | None = None,
    tau_mouse_scale: float = 1.0,
    tau_human_scale: float = 1.0,
) -> np.ndarray:
    """Return a full (n_mouse, n_human) coupling for one anchor-only baseline."""
    idx_m = get_anchor_index(mouse_ad.var)
    idx_h = get_anchor_index(human_ad.var)
    n_m = int(mouse_ad.uns["n_nodes"])
    n_h = int(human_ad.uns["n_nodes"])

    visible_local = _visible_anchor_indices(idx_m, visible_pair_ids)
    if visible_local.size == 0:
        return np.full((n_m, n_h), 1.0 / (n_m * n_h), dtype=np.float64)

    m_xyz = _xyz(mouse_ad.var)
    h_xyz = _xyz(human_ad.var)
    m_anchor_pos = idx_m.pos[visible_local]
    h_anchor_pos = idx_h.pos[visible_local]
    m_anchor_xyz = m_xyz[m_anchor_pos]
    h_anchor_xyz = h_xyz[h_anchor_pos]

    if model == "uniform":
        return np.full((n_m, n_h), 1.0 / (n_m * n_h), dtype=np.float64)

    if model == "visible_anchor_prior":
        row_prob = np.zeros((n_m, n_h), dtype=np.float64)
        row_prob[:, h_anchor_pos] = 1.0 / float(len(h_anchor_pos))
        return _row_scaled_coupling(row_prob)

    if tau_mouse is None:
        tau_mouse = _auto_tau(m_anchor_xyz) * float(tau_mouse_scale)
    if tau_human is None:
        tau_human = _auto_tau(h_anchor_xyz) * float(tau_human_scale)

    d_mouse = _pairwise_dist(m_xyz, m_anchor_xyz)
    if model == "nearest_anchor_delta":
        nearest = np.argmin(d_mouse, axis=1)
        row_prob = np.zeros((n_m, n_h), dtype=np.float64)
        row_prob[np.arange(n_m), h_anchor_pos[nearest]] = 1.0
        return _row_scaled_coupling(row_prob)

    w_mouse = np.exp(-0.5 * (d_mouse / max(tau_mouse, 1e-9)) ** 2)
    w_mouse = _normalise_rows(w_mouse)

    if model == "mouse_kernel_delta":
        row_prob = np.zeros((n_m, n_h), dtype=np.float64)
        for k, hp in enumerate(h_anchor_pos):
            row_prob[:, hp] += w_mouse[:, k]
        return _row_scaled_coupling(row_prob)

    if model == "mouse_kernel_human_kernel":
        d_human = _pairwise_dist(h_anchor_xyz, h_xyz)
        human_kernels = np.exp(-0.5 * (d_human / max(tau_human, 1e-9)) ** 2)
        human_kernels = _normalise_rows(human_kernels)
        row_prob = w_mouse @ human_kernels
        row_prob = _normalise_rows(row_prob)
        return _row_scaled_coupling(row_prob)

    raise ValueError(f"unknown baseline model: {model}")


def _network_to_pair_ids() -> dict[str, list[int]]:
    out = {n: [] for n in NETWORKS}
    for pid, net_name in PAIRID_TO_NETWORK.items():
        out[net_name].append(int(pid))
    return out


def _weighted_summary(per_network: dict[str, dict]) -> dict:
    keys = [
        "top1", "top5", "pair_id", "hemisphere", "mean_rank", "median_rank",
        "mean_xyz_dist", "full_top1", "full_top5", "mean_rank_full",
        "mean_mass_on_correct_anchor", "frac_in_neighborhood",
    ]
    weights = np.array([v["n_anchors_held"] for v in per_network.values()], dtype=float)
    out = {"n_anchors": int(weights.sum())}
    if weights.sum() <= 0:
        return out
    for key in keys:
        vals = np.array([v.get(key, np.nan) for v in per_network.values()], dtype=float)
        ok = np.isfinite(vals)
        if ok.any():
            out[key] = float(np.sum(vals[ok] * weights[ok]) / np.sum(weights[ok]))
    return out


def _restricted_per_anchor_rows(
    pi_anchor: np.ndarray,
    idx_m,
    idx_h,
    held_out_pair_ids: Iterable[int],
    *,
    var_h=None,
) -> list[dict]:
    """Per-anchor version of held_out_metrics_graded for bootstrap CIs."""
    from homer.data.anchors import held_out_indices

    held_set = set(int(x) for x in held_out_pair_ids)
    visible_pair_ids = [p for p in idx_m.pair_ids if int(p) not in held_set]
    m_held, h_held = held_out_indices(idx_m, idx_h, visible_pair_ids)
    if len(m_held) == 0:
        return []

    sub = pi_anchor[np.ix_(m_held, h_held)]
    order = np.argsort(-sub, axis=1)
    pred_h_local = order[:, 0]
    pred_h_global = h_held[pred_h_local]
    true_local = np.arange(len(m_held))

    xyz_n = None
    if var_h is not None:
        xyz = var_h[["x", "y", "z"]].values.astype(np.float64)
        lo = xyz.min(0, keepdims=True)
        hi = xyz.max(0, keepdims=True)
        xyz_n = (xyz - lo) / np.maximum(hi - lo, 1e-9)
        anchor_xyz = xyz_n[idx_h.pos]

    rows: list[dict] = []
    for i, ml in enumerate(m_held):
        rank = int(np.where(order[i] == true_local[i])[0][0]) + 1
        row = {
            "pair_id": int(idx_m.pair_ids[ml]),
            "hemisphere_label": str(idx_m.hemispheres[ml]),
            "top1": float(pred_h_global[i] == ml),
            "top5": float(rank <= 5),
            "pair_id_accuracy": float(idx_m.pair_ids[ml] == idx_h.pair_ids[pred_h_global[i]]),
            "hemisphere": float(idx_m.hemispheres[ml] == idx_h.hemispheres[pred_h_global[i]]),
            "rank": float(rank),
        }
        if xyz_n is not None:
            true_xyz = anchor_xyz[h_held[true_local[i]]]
            pred_xyz = anchor_xyz[pred_h_global[i]]
            row["xyz_dist"] = float(np.linalg.norm(true_xyz - pred_xyz))
        rows.append(row)
    return rows


def _full_per_anchor_rows(
    pi: np.ndarray,
    idx_m,
    idx_h,
    held_out_pair_ids: Iterable[int],
    *,
    var_h=None,
    top_k: int = 5,
    neighborhood_xyz_dist: float = 0.05,
) -> list[dict]:
    """Per-anchor full-space rows mirroring full_space_metrics."""
    from homer.data.anchors import held_out_indices

    held_set = set(int(x) for x in held_out_pair_ids)
    visible_pair_ids = [p for p in idx_m.pair_ids if int(p) not in held_set]
    m_held_local, h_held_local = held_out_indices(idx_m, idx_h, visible_pair_ids)
    if len(m_held_local) == 0:
        return []

    m_held_pos = idx_m.pos[m_held_local]
    h_correct_pos = idx_h.pos[h_held_local]
    sub_full = pi[m_held_pos, :]
    order = np.argsort(-sub_full, axis=1)
    full_argmax = order[:, 0]

    xyz_n = None
    if var_h is not None:
        xyz = var_h[["x", "y", "z"]].values.astype(np.float64)
        lo = xyz.min(0, keepdims=True)
        hi = xyz.max(0, keepdims=True)
        xyz_n = (xyz - lo) / np.maximum(hi - lo, 1e-9)

    rows: list[dict] = []
    for i, ml in enumerate(m_held_local):
        rank = int(np.where(order[i] == h_correct_pos[i])[0][0]) + 1
        row_sum = float(sub_full[i].sum())
        row = {
            "pair_id": int(idx_m.pair_ids[ml]),
            "hemisphere_label": str(idx_m.hemispheres[ml]),
            "full_top1": float(full_argmax[i] == h_correct_pos[i]),
            f"full_top{top_k}": float(rank <= top_k),
            "rank_full": float(rank),
            "mass_on_correct_anchor": float(sub_full[i, h_correct_pos[i]] / max(row_sum, 1e-12)),
        }
        if xyz_n is not None:
            dist = float(np.linalg.norm(xyz_n[full_argmax[i]] - xyz_n[h_correct_pos[i]]))
            row["xyz_dist_full"] = dist
            row["in_neighborhood"] = float(dist < neighborhood_xyz_dist)
        rows.append(row)
    return rows


def _merge_per_anchor_rows(restricted: list[dict], full: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for r, f in zip(restricted, full, strict=True):
        row = dict(r)
        row.update(f)
        merged.append(row)
    return merged


def _summarise_anchor_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"n_anchors": 0}
    specs = {
        "top1": "top1",
        "top5": "top5",
        "pair_id": "pair_id_accuracy",
        "hemisphere": "hemisphere",
        "mean_rank": "rank",
        "mean_xyz_dist": "xyz_dist",
        "full_top1": "full_top1",
        "full_top5": "full_top5",
        "mean_rank_full": "rank_full",
        "mean_mass_on_correct_anchor": "mass_on_correct_anchor",
        "frac_in_neighborhood": "in_neighborhood",
    }
    out = {"n_anchors": len(rows)}
    for out_key, row_key in specs.items():
        vals = np.array([r.get(row_key, np.nan) for r in rows], dtype=float)
        ok = np.isfinite(vals)
        if ok.any():
            out[out_key] = float(vals[ok].mean())
    return out


def _bootstrap_ci(
    rows: list[dict],
    *,
    n_boot: int = 2000,
    seed: int = 123,
    alpha: float = 0.05,
) -> dict:
    if not rows or n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(rows)
    keys = [
        "top1", "top5", "pair_id", "hemisphere", "mean_rank", "mean_xyz_dist",
        "full_top1", "full_top5", "mean_rank_full", "mean_mass_on_correct_anchor",
        "frac_in_neighborhood",
    ]
    draws = {key: [] for key in keys}
    for _ in range(n_boot):
        sample = [rows[i] for i in rng.integers(0, n, size=n)]
        summary = _summarise_anchor_rows(sample)
        for key in keys:
            if key in summary and math.isfinite(float(summary[key])):
                draws[key].append(float(summary[key]))

    out = {}
    lo_q = 100 * alpha / 2.0
    hi_q = 100 * (1.0 - alpha / 2.0)
    for key, vals in draws.items():
        if vals:
            arr = np.array(vals, dtype=float)
            out[key] = {
                "mean": float(arr.mean()),
                "lo": float(np.percentile(arr, lo_q)),
                "hi": float(np.percentile(arr, hi_q)),
                "n_boot": int(len(arr)),
            }
    return out


def evaluate_lono_baselines(
    mouse_ad,
    human_ad,
    *,
    models: list[str],
    networks: list[str],
    tau_mouse: float | None = None,
    tau_human: float | None = None,
    model_tau_configs: dict[str, dict] | None = None,
    include_fc_translation: bool = True,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 123,
    keep_per_anchor: bool = True,
) -> dict:
    idx_m = get_anchor_index(mouse_ad.var)
    idx_h = get_anchor_index(human_ad.var)
    net_to_pairs = _network_to_pair_ids()
    all_pair_ids = set(int(p) for p in idx_m.pair_ids)

    out: dict[str, dict] = {}
    for model in models:
        per_network: dict[str, dict] = {}
        per_anchor_all: list[dict] = []
        cfg = (model_tau_configs or {}).get(model, {})
        model_tau_mouse = cfg.get("tau_mouse", tau_mouse)
        model_tau_human = cfg.get("tau_human", tau_human)
        model_tau_mouse_scale = cfg.get("tau_mouse_scale", 1.0)
        model_tau_human_scale = cfg.get("tau_human_scale", 1.0)
        for net_name in networks:
            held = sorted(net_to_pairs[net_name])
            visible = sorted(all_pair_ids - set(held))
            pi = build_anchor_baseline(
                model, mouse_ad, human_ad, visible_pair_ids=visible,
                tau_mouse=model_tau_mouse, tau_human=model_tau_human,
                tau_mouse_scale=model_tau_mouse_scale,
                tau_human_scale=model_tau_human_scale,
            )
            pi_anchor = pi[np.ix_(idx_m.pos, idx_h.pos)]
            restricted = held_out_metrics_graded(
                pi_anchor, idx_m, idx_h, held, var_h=human_ad.var,
            )
            full = full_space_metrics(
                pi, idx_m, idx_h, held, var_h=human_ad.var, top_k=5,
            )
            row = {
                "n_anchors_held": restricted["n"],
                "n_pair_ids_held": len(held),
                **restricted,
                "hemi": restricted.get("hemisphere", float("nan")),
            }
            for key, value in full.items():
                if key != "n":
                    row[key] = value
            per_network[net_name] = row

            anchor_rows = _merge_per_anchor_rows(
                _restricted_per_anchor_rows(
                    pi_anchor, idx_m, idx_h, held, var_h=human_ad.var,
                ),
                _full_per_anchor_rows(
                    pi, idx_m, idx_h, held, var_h=human_ad.var, top_k=5,
                ),
            )
            for anchor_row in anchor_rows:
                anchor_row["network"] = net_name
                anchor_row["model"] = model
            per_anchor_all.extend(anchor_rows)

        model_out = {
            "per_network": per_network,
            "weighted": _weighted_summary(per_network),
            "bootstrap_ci": _bootstrap_ci(
                per_anchor_all, n_boot=n_bootstrap, seed=bootstrap_seed,
            ),
            "tau_config": cfg,
        }
        if keep_per_anchor:
            model_out["per_anchor"] = per_anchor_all

        if include_fc_translation:
            # For FC translation use all anchors visible, because this is not a
            # held-out-anchor fold metric.
            pi_all = build_anchor_baseline(
                model, mouse_ad, human_ad, visible_pair_ids=all_pair_ids,
                tau_mouse=model_tau_mouse, tau_human=model_tau_human,
                tau_mouse_scale=model_tau_mouse_scale,
                tau_human_scale=model_tau_human_scale,
            )
            try:
                net_h = assign_networks(human_ad.var, idx_h)
                model_out["fc_translation_all_anchors"] = fc_translation_quality(
                    pi_all,
                    mouse_ad.uns["fc_mean"].astype(np.float64),
                    human_ad.uns["fc_mean"].astype(np.float64),
                    network_labels_h=net_h,
                )
            except Exception as exc:  # keep anchor metrics if FC eval fails
                model_out["fc_translation_all_anchors_error"] = str(exc)

        out[model] = model_out
    return out


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _score_for_tuning(weighted: dict) -> tuple[float, float, float]:
    """Maximise top1, then minimise mean rank and xyz distance."""
    top1 = float(weighted.get("top1", float("-inf")))
    mean_rank = float(weighted.get("mean_rank", float("inf")))
    mean_xyz = float(weighted.get("mean_xyz_dist", float("inf")))
    return (top1, -mean_rank, -mean_xyz)


def tune_bandwidths(
    mouse_ad,
    human_ad,
    *,
    models: list[str],
    networks: list[str],
    tau_grid: list[float],
) -> dict:
    """Tune kernel bandwidth multipliers using LONO anchor metrics only."""
    kernel_models = {"mouse_kernel_delta", "mouse_kernel_human_kernel"}
    out: dict[str, dict] = {}
    for model in models:
        if model not in kernel_models:
            continue

        candidates: list[dict] = []
        if model == "mouse_kernel_delta":
            grid = [(tm, 1.0) for tm in tau_grid]
        else:
            grid = [(tm, th) for tm in tau_grid for th in tau_grid]

        best: dict | None = None
        for tau_mouse_scale, tau_human_scale in grid:
            cfg = {
                model: {
                    "tau_mouse_scale": float(tau_mouse_scale),
                    "tau_human_scale": float(tau_human_scale),
                }
            }
            evaluated = evaluate_lono_baselines(
                mouse_ad,
                human_ad,
                models=[model],
                networks=networks,
                model_tau_configs=cfg,
                include_fc_translation=False,
                n_bootstrap=0,
                keep_per_anchor=False,
            )[model]
            weighted = evaluated["weighted"]
            candidate = {
                "tau_mouse_scale": float(tau_mouse_scale),
                "tau_human_scale": float(tau_human_scale),
                "weighted": weighted,
            }
            candidates.append(candidate)
            if best is None or _score_for_tuning(weighted) > _score_for_tuning(best["weighted"]):
                best = candidate

        out[model] = {
            "selection_metric": "max top1, then min mean_rank, then min mean_xyz_dist",
            "selected": best,
            "candidates": candidates,
        }
    return out


def selected_tau_configs(tuning: dict) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for model, payload in tuning.items():
        selected = payload.get("selected")
        if selected:
            configs[model] = {
                "tau_mouse_scale": selected["tau_mouse_scale"],
                "tau_human_scale": selected["tau_human_scale"],
            }
    return configs


def _weighted_from_cv_log(payload: dict) -> dict | None:
    """Extract a weighted summary from common HOMER LONO log shapes."""
    if not payload:
        return None
    if "weighted" in payload and isinstance(payload["weighted"], dict):
        return payload["weighted"]
    if "per_network" in payload and isinstance(payload["per_network"], dict):
        rows = payload["per_network"]
    else:
        rows = {k: v for k, v in payload.items()
                if isinstance(v, dict) and "top1" in v}
    if not rows:
        return None
    normalised = {}
    for name, row in rows.items():
        normalised[name] = {
            "n_anchors_held": row.get("n_anchors_held", row.get("n", 0)),
            "top1": row.get("top1", np.nan),
            "top5": row.get("top5", np.nan),
            "pair_id": row.get("pair_id", np.nan),
            "hemisphere": row.get("hemisphere", row.get("hemi", np.nan)),
            "mean_rank": row.get("mean_rank", np.nan),
            "mean_xyz_dist": row.get("mean_xyz_dist", np.nan),
        }
    return _weighted_summary(normalised)


def collect_existing_homer_logs() -> dict:
    """Collect comparable HOMER metrics already committed under outputs/logs."""
    out: dict[str, dict] = {}

    multimodal = _load_json(LOGS / "multimodal_cv.json")
    if multimodal:
        for key in ("baseline_fc_only", "fc_plus_SC"):
            if key in multimodal:
                out[f"homer_lono_{key}"] = {
                    "source": "outputs/logs/multimodal_cv.json",
                    "weighted": _weighted_from_cv_log(multimodal[key]),
                }

    for name, file_name in [
        ("homer_lono_no_xyz", "garin_supervised_cv_no_xyz.json"),
        ("homer_lono_fc_only_summary", "garin_supervised_cv.json"),
        ("homer_full_space_eval", "full_space_eval.json"),
        ("homer_fc_translation", "fc_translation.json"),
        ("homer_null_distributions", "null_distributions.json"),
    ]:
        payload = _load_json(LOGS / file_name)
        if payload is not None:
            item: dict = {"source": f"outputs/logs/{file_name}"}
            weighted = _weighted_from_cv_log(payload)
            if weighted is not None:
                item["weighted"] = weighted
            else:
                item["raw_keys"] = sorted(payload.keys())[:20]
            out[name] = item
    return out


def write_summary(result: dict, path: Path) -> None:
    lines = [
        "# Anchor-interpolation baseline results",
        "",
        f"Status: **{result['status']}**",
        "",
    ]
    if result.get("reason"):
        lines += [result["reason"], ""]

    baselines = result.get("anchor_interpolation_baselines", {})
    if baselines:
        lines += [
            "## Weighted Leave-One-Network-Out Metrics",
            "",
            "| model | tau scales | top1 (95% CI) | top5 | mean rank (95% CI) | full top5 | FC translation r |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, payload in baselines.items():
            w = payload.get("weighted", {})
            ci = payload.get("bootstrap_ci", {})
            fc = payload.get("fc_translation_all_anchors", {})
            tau = payload.get("tau_config", {})
            tau_label = _fmt_tau(tau)
            lines.append(
                f"| `{name}` | {tau_label} | {_fmt_pct_ci(w.get('top1'), ci.get('top1'))} | "
                f"{_fmt_pct(w.get('top5'))} | {_fmt_num_ci(w.get('mean_rank'), ci.get('mean_rank'))} | "
                f"{_fmt_pct(w.get('full_top5'))} | {_fmt_num(fc.get('pearson_r_overall'))} |"
            )
        lines.append("")

    tuning = result.get("bandwidth_tuning", {})
    if tuning:
        lines += ["## Bandwidth Tuning", ""]
        for model, payload in tuning.items():
            selected = payload.get("selected", {})
            w = selected.get("weighted", {})
            lines.append(
                f"- `{model}` selected tau_mouse_scale={_fmt_num(selected.get('tau_mouse_scale'))}, "
                f"tau_human_scale={_fmt_num(selected.get('tau_human_scale'))}; "
                f"top1={_fmt_pct(w.get('top1'))}, mean_rank={_fmt_num(w.get('mean_rank'))}"
            )
        lines.append("")

    homer = result.get("existing_homer_logs", {})
    if homer:
        lines += ["## Existing HOMER Log Summaries", ""]
        for name, payload in homer.items():
            lines.append(f"- `{name}` from `{payload.get('source')}`")
            if payload.get("weighted"):
                w = payload["weighted"]
                bits = []
                for key in ("top1", "top5", "pair_id", "mean_rank", "mean_xyz_dist"):
                    if key in w:
                        bits.append(f"{key}={_fmt_pct(w[key]) if key in ('top1', 'top5', 'pair_id') else _fmt_num(w[key])}")
                if bits:
                    lines.append(f"  {', '.join(bits)}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def _fmt_pct(x) -> str:
    try:
        if x is None or not math.isfinite(float(x)):
            return "n/a"
        return f"{100 * float(x):.1f}%"
    except Exception:
        return "n/a"


def _fmt_num(x) -> str:
    try:
        if x is None or not math.isfinite(float(x)):
            return "n/a"
        return f"{float(x):.3g}"
    except Exception:
        return "n/a"


def _fmt_pct_ci(x, ci) -> str:
    base = _fmt_pct(x)
    if not ci:
        return base
    return f"{base} [{_fmt_pct(ci.get('lo'))}, {_fmt_pct(ci.get('hi'))}]"


def _fmt_num_ci(x, ci) -> str:
    base = _fmt_num(x)
    if not ci:
        return base
    return f"{base} [{_fmt_num(ci.get('lo'))}, {_fmt_num(ci.get('hi'))}]"


def _fmt_tau(tau: dict) -> str:
    if not tau:
        return "n/a"
    tm = tau.get("tau_mouse_scale")
    th = tau.get("tau_human_scale")
    if tm is None and th is None:
        return "n/a"
    return f"m={_fmt_num(tm)}, h={_fmt_num(th)}"


def _parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="uniform,visible_anchor_prior,nearest_anchor_delta,mouse_kernel_delta,mouse_kernel_human_kernel",
                    help="comma-separated anchor baselines to run")
    ap.add_argument("--networks", default=None,
                    help="comma-separated network subset; default all")
    ap.add_argument("--tau-mouse", type=float, default=None,
                    help="mouse kernel bandwidth in mm; default median visible-anchor NN distance per fold")
    ap.add_argument("--tau-human", type=float, default=None,
                    help="human kernel bandwidth in mm; default median visible-anchor NN distance per fold")
    ap.add_argument("--skip-fc-translation", action="store_true",
                    help="skip expensive FC translation metric")
    ap.add_argument("--tune-bandwidths", action="store_true",
                    help="grid-search kernel bandwidth multipliers before final evaluation")
    ap.add_argument("--tau-grid", default="0.25,0.5,1,2,4,8",
                    help="comma-separated bandwidth multipliers for --tune-bandwidths")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="number of per-anchor bootstrap resamples for confidence intervals")
    ap.add_argument("--bootstrap-seed", type=int, default=123,
                    help="random seed for bootstrap confidence intervals")
    args = ap.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS / "anchor_interpolation_baseline.json"
    summary_path = RESULTS / "summary.md"

    result: dict = {
        "status": "started",
        "experiment": "anchor_interpolation_baseline",
        "models": args.models.split(","),
        "networks": args.networks.split(",") if args.networks else list(NETWORKS),
        "bootstrap_resamples": int(args.bootstrap),
        "existing_homer_logs": collect_existing_homer_logs(),
    }

    try:
        human_ad, _ = load_cached("human", cache_dir=ANN)
        mouse_ad, _ = load_cached("mouse", cache_dir=ANN)
    except DataNotFound as exc:
        result["status"] = "data_missing"
        result["reason"] = (
            "The anchor-only baselines require the fetched AnnData caches. "
            "Run `python scripts/fetch_data.py` from the repository root, then "
            "rerun this script. Original error:\n\n"
            f"```text\n{exc}\n```"
        )
        result_path.write_text(json.dumps(result, indent=2, default=float))
        write_summary(result, summary_path)
        print(result["reason"])
        print(f"\nwrote → {result_path}")
        print(f"wrote → {summary_path}")
        return 2

    model_tau_configs = {}
    if args.tune_bandwidths:
        tau_grid = _parse_float_list(args.tau_grid)
        result["tau_grid"] = tau_grid
        result["bandwidth_tuning"] = tune_bandwidths(
            mouse_ad,
            human_ad,
            models=result["models"],
            networks=result["networks"],
            tau_grid=tau_grid,
        )
        model_tau_configs = selected_tau_configs(result["bandwidth_tuning"])

    result["anchor_interpolation_baselines"] = evaluate_lono_baselines(
        mouse_ad,
        human_ad,
        models=result["models"],
        networks=result["networks"],
        tau_mouse=args.tau_mouse,
        tau_human=args.tau_human,
        model_tau_configs=model_tau_configs,
        include_fc_translation=not args.skip_fc_translation,
        n_bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["status"] = "complete"
    result_path.write_text(json.dumps(result, indent=2, default=float))
    write_summary(result, summary_path)
    print(f"wrote → {result_path}")
    print(f"wrote → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
