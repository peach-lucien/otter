"""Generate plots for the anchor-interpolation baseline experiment."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
JSON_PATH = RESULTS / "anchor_interpolation_baseline.json"
ROOT = HERE.parents[1]
LOGS = ROOT / "outputs" / "logs"

MODEL_LABELS = {
    "uniform": "Uniform",
    "visible_anchor_prior": "Visible\nanchors",
    "nearest_anchor_delta": "Nearest\nanchor",
    "mouse_kernel_delta": "Mouse\nkernel",
    "mouse_kernel_human_kernel": "Mouse+human\nkernel",
}

COLORS = {
    "uniform": "#8a8f98",
    "visible_anchor_prior": "#5f7f95",
    "nearest_anchor_delta": "#3f6c51",
    "mouse_kernel_delta": "#9a6b35",
    "mouse_kernel_human_kernel": "#7b4f8f",
}

HOMER_COLORS = {
    "HOMER FC-only": "#222222",
    "HOMER FC+SC": "#555555",
    "HOMER FC+xyz/GW": "#777777",
    "HOMER selected": "#0f766e",
}


def load_result() -> dict:
    return json.loads(JSON_PATH.read_text())


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _models(result: dict) -> list[str]:
    return list(result["anchor_interpolation_baselines"].keys())


def _weighted(result: dict, model: str, key: str) -> float:
    return float(result["anchor_interpolation_baselines"][model]["weighted"].get(key, np.nan))


def _ci(result: dict, model: str, key: str) -> tuple[float, float] | None:
    ci = result["anchor_interpolation_baselines"][model].get("bootstrap_ci", {}).get(key)
    if not ci:
        return None
    return float(ci["lo"]), float(ci["hi"])


def _save(fig, name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{name}.png", dpi=180)
    fig.savefig(PLOTS / f"{name}.svg")
    plt.close(fig)


def _weighted_log_summary(rows: dict, key: str) -> float:
    vals = []
    weights = []
    for row in rows.values():
        if isinstance(row, dict) and key in row and "n" in row:
            vals.append(float(row[key]))
            weights.append(float(row["n"]))
    if not weights or sum(weights) <= 0:
        return float("nan")
    vals_arr = np.array(vals, dtype=float)
    weights_arr = np.array(weights, dtype=float)
    return float(np.sum(vals_arr * weights_arr) / np.sum(weights_arr))


def _homer_full_space_refs() -> dict[str, dict]:
    payload = _load_json(LOGS / "full_space_eval.json")
    rows = payload.get("fc_plus_SC", {})
    if not rows:
        return {}
    return {
        "HOMER FC+SC": {
            "full_top5": _weighted_log_summary(rows, "full_top5"),
            "frac_in_neighborhood": _weighted_log_summary(rows, "frac_in_neighborhood"),
        }
    }


def _homer_fc_refs() -> dict[str, float]:
    payload = _load_json(LOGS / "fc_translation.json")
    mapping = {
        "HOMER FC-only": "baseline_fc_only",
        "HOMER FC+SC": "fc_plus_SC",
        "HOMER FC+xyz/GW": "fc_plus_xyz_gw",
    }
    out = {}
    for label, key in mapping.items():
        val = payload.get(key, {}).get("pearson_r_overall")
        if val is not None and np.isfinite(float(val)):
            out[label] = float(val)
    return out


def _selected_weight_metrics() -> dict:
    payload = _load_json(LOGS / "weight_selection_selected.json")
    return payload.get("metrics", {})


def plot_anchor_accuracy(result: dict) -> None:
    models = _models(result)
    x = np.arange(len(models))
    top1 = np.array([_weighted(result, m, "top1") for m in models])
    top5 = np.array([_weighted(result, m, "top5") for m in models])
    yerr = np.zeros((2, len(models)))
    for i, m in enumerate(models):
        ci = _ci(result, m, "top1")
        if ci:
            yerr[0, i] = top1[i] - ci[0]
            yerr[1, i] = ci[1] - top1[i]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - 0.18, top1 * 100, width=0.36, yerr=yerr * 100, capsize=4,
           color=[COLORS[m] for m in models], label="Top-1")
    ax.bar(x + 0.18, top5 * 100, width=0.36, color="#c9cdd2", label="Top-5")
    ax.axhline(78.6, color="#222222", linewidth=1.2, linestyle="--", label="HOMER FC-only top-1")
    ax.axhline(81.0, color="#222222", linewidth=1.2, linestyle=":", label="HOMER FC+SC top-1")
    selected = _selected_weight_metrics()
    if selected:
        ax.axhline(
            100 * selected["anchor_top1"],
            color=HOMER_COLORS["HOMER selected"],
            linewidth=1.4,
            linestyle="-.",
            label="HOMER selected top-1",
        )
    ax.set_ylim(0, 108)
    ax.set_ylabel("Held-out anchor accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_title("Restricted held-out anchor ranking")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "anchor_accuracy_with_ci")


def plot_rank_and_distance(result: dict) -> None:
    models = _models(result)
    x = np.arange(len(models))
    rank = np.array([_weighted(result, m, "mean_rank") for m in models])
    dist = np.array([_weighted(result, m, "mean_xyz_dist") for m in models])
    rank_err = np.zeros((2, len(models)))
    for i, m in enumerate(models):
        ci = _ci(result, m, "mean_rank")
        if ci:
            rank_err[0, i] = rank[i] - ci[0]
            rank_err[1, i] = ci[1] - rank[i]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    axes[0].bar(x, rank, yerr=rank_err, capsize=4, color=[COLORS[m] for m in models])
    axes[0].axhline(1.26, color="#222222", linewidth=1.2, linestyle="--", label="HOMER FC-only")
    axes[0].axhline(1.24, color="#222222", linewidth=1.2, linestyle=":", label="HOMER FC+SC")
    selected = _selected_weight_metrics()
    if selected:
        axes[0].axhline(
            selected["anchor_mean_rank"],
            color=HOMER_COLORS["HOMER selected"],
            linewidth=1.4,
            linestyle="-.",
            label="HOMER selected",
        )
    axes[0].set_ylabel("Mean rank, lower is better")
    axes[0].set_title("Correct anchor rank")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([MODEL_LABELS[m] for m in models], rotation=0)
    axes[0].legend(frameon=False)

    axes[1].bar(x, dist, color=[COLORS[m] for m in models])
    axes[1].axhline(0.0212, color="#222222", linewidth=1.2, linestyle="--", label="HOMER FC-only")
    axes[1].axhline(0.0200, color="#222222", linewidth=1.2, linestyle=":", label="HOMER FC+SC")
    if selected:
        axes[1].axhline(
            selected["anchor_mean_xyz_dist"],
            color=HOMER_COLORS["HOMER selected"],
            linewidth=1.4,
            linestyle="-.",
            label="HOMER selected",
        )
    axes[1].set_ylabel("Normalised xyz distance, lower is better")
    axes[1].set_title("Distance from predicted to true anchor")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODEL_LABELS[m] for m in models], rotation=0)
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "rank_and_xyz_distance")


def plot_full_space_and_fc(result: dict) -> None:
    models = _models(result)
    x = np.arange(len(models))
    full_top5 = np.array([_weighted(result, m, "full_top5") for m in models])
    neighborhood = np.array([_weighted(result, m, "frac_in_neighborhood") for m in models])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes[0].bar(x - 0.18, full_top5 * 100, width=0.36, color=[COLORS[m] for m in models], label="Full top-5")
    axes[0].bar(x + 0.18, neighborhood * 100, width=0.36, color="#c9cdd2", label="Argmax near true anchor")
    homer_full = _homer_full_space_refs()
    if homer_full:
        axes[0].axhline(
            100 * homer_full["HOMER FC+SC"]["full_top5"],
            color="#222222",
            linewidth=1.2,
            linestyle=":",
            label="HOMER FC+SC full top-5",
        )
    axes[0].set_ylabel("Full-space recovery (%)")
    axes[0].set_title("Harder all-human-parcel search")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([MODEL_LABELS[m] for m in models])
    axes[0].legend(frameon=False)

    # FC translation is only directly comparable when evaluated over the
    # brain-wide human parcel space. The delta anchor baselines keep only 42
    # human anchors, so they are intentionally omitted from this panel.
    comparable = ["mouse_kernel_human_kernel"]
    labels = ["Tuned anchor\nspatial"]
    vals = [
        float(result["anchor_interpolation_baselines"][comparable[0]]
              ["fc_translation_all_anchors"]["pearson_r_overall"])
    ]
    colors = [COLORS[comparable[0]]]
    selected = _selected_weight_metrics()
    if selected:
        labels.append("HOMER\nselected")
        vals.append(float(selected["fc_translation_r"]))
        colors.append(HOMER_COLORS["HOMER selected"])
    for label, val in _homer_fc_refs().items():
        labels.append(label.replace(" ", "\n", 1))
        vals.append(val)
        colors.append(HOMER_COLORS[label])

    x_fc = np.arange(len(vals))
    axes[1].bar(x_fc, vals, color=colors)
    axes[1].set_ylim(0, max(0.5, max(vals) + 0.08))
    axes[1].set_ylabel("Pearson r")
    axes[1].set_title("Brain-wide FC translation")
    axes[1].set_xticks(x_fc)
    axes[1].set_xticklabels(labels)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "full_space_and_fc_translation")


def plot_fc_translation_breakdown(result: dict) -> None:
    """Compare brain-wide FC preservation overall, within-network, and cross-network."""
    anchor_fc = result["anchor_interpolation_baselines"]["mouse_kernel_human_kernel"][
        "fc_translation_all_anchors"
    ]
    fc_log = _load_json(LOGS / "fc_translation.json")
    selected = _selected_weight_metrics()
    rows = [
        (
            "Tuned anchor\nspatial",
            anchor_fc.get("pearson_r_overall"),
            anchor_fc.get("pearson_r_within_net"),
            anchor_fc.get("pearson_r_cross_net"),
            COLORS["mouse_kernel_human_kernel"],
        ),
    ]
    if selected:
        rows.append((
            "HOMER\nselected",
            selected.get("fc_translation_r"),
            selected.get("fc_within_r"),
            selected.get("fc_cross_r"),
            HOMER_COLORS["HOMER selected"],
        ))
    rows += [
        (
            "HOMER\nFC-only",
            fc_log.get("baseline_fc_only", {}).get("pearson_r_overall"),
            fc_log.get("baseline_fc_only", {}).get("pearson_r_within_net"),
            fc_log.get("baseline_fc_only", {}).get("pearson_r_cross_net"),
            HOMER_COLORS["HOMER FC-only"],
        ),
        (
            "HOMER\nFC+SC",
            fc_log.get("fc_plus_SC", {}).get("pearson_r_overall"),
            fc_log.get("fc_plus_SC", {}).get("pearson_r_within_net"),
            fc_log.get("fc_plus_SC", {}).get("pearson_r_cross_net"),
            HOMER_COLORS["HOMER FC+SC"],
        ),
        (
            "HOMER\nFC+xyz/GW",
            fc_log.get("fc_plus_xyz_gw", {}).get("pearson_r_overall"),
            fc_log.get("fc_plus_xyz_gw", {}).get("pearson_r_within_net"),
            fc_log.get("fc_plus_xyz_gw", {}).get("pearson_r_cross_net"),
            HOMER_COLORS["HOMER FC+xyz/GW"],
        ),
    ]
    labels = [r[0] for r in rows]
    vals = np.array([[float(v) for v in r[1:4]] for r in rows], dtype=float)
    colors = [r[4] for r in rows]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(x - width, vals[:, 0], width=width, color=colors, alpha=0.95, label="Overall")
    ax.bar(x, vals[:, 1], width=width, color=colors, alpha=0.55, label="Within-network")
    ax.bar(x + width, vals[:, 2], width=width, color=colors, alpha=0.25, label="Cross-network")
    ax.set_ylim(0, max(0.6, float(np.nanmax(vals)) + 0.08))
    ax.set_ylabel("Pearson r")
    ax.set_title("Brain-wide FC translation breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fc_translation_breakdown")


def plot_per_network(result: dict) -> None:
    models = _models(result)
    networks = result["networks"]
    mat = np.array([
        [
            result["anchor_interpolation_baselines"][m]["per_network"][net]["top1"] * 100
            for net in networks
        ]
        for m in models
    ])
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    im = ax.imshow(mat, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([MODEL_LABELS[m].replace("\n", " ") for m in models])
    ax.set_xticks(np.arange(len(networks)))
    ax.set_xticklabels(networks, rotation=35, ha="right")
    ax.set_title("Top-1 accuracy by held-out network")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Top-1 (%)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    color="white" if mat[i, j] < 55 else "black", fontsize=8)
    _save(fig, "per_network_top1_heatmap")


def plot_tuning_surface(result: dict) -> None:
    tuning = result.get("bandwidth_tuning", {}).get("mouse_kernel_human_kernel")
    if not tuning:
        return
    candidates = tuning["candidates"]
    xs = sorted({float(c["tau_mouse_scale"]) for c in candidates})
    ys = sorted({float(c["tau_human_scale"]) for c in candidates})
    grid = np.full((len(ys), len(xs)), np.nan)
    for c in candidates:
        i = ys.index(float(c["tau_human_scale"]))
        j = xs.index(float(c["tau_mouse_scale"]))
        grid[i, j] = 100 * float(c["weighted"]["top1"])

    selected = tuning["selected"]
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    im = ax.imshow(grid, vmin=0, vmax=100, cmap="magma", origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([str(x).rstrip("0").rstrip(".") for x in xs])
    ax.set_yticks(np.arange(len(ys)))
    ax.set_yticklabels([str(y).rstrip("0").rstrip(".") for y in ys])
    ax.set_xlabel("Mouse bandwidth scale")
    ax.set_ylabel("Human bandwidth scale")
    ax.set_title("Bandwidth tuning: mouse+human kernel top-1")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Top-1 (%)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                    color="white" if grid[i, j] < 55 else "black", fontsize=8)
    sx = xs.index(float(selected["tau_mouse_scale"]))
    sy = ys.index(float(selected["tau_human_scale"]))
    ax.scatter([sx], [sy], marker="s", s=180, facecolors="none", edgecolors="#19d3c5", linewidths=2.5)
    _save(fig, "bandwidth_tuning_surface")


def main() -> int:
    result = load_result()
    plot_anchor_accuracy(result)
    plot_rank_and_distance(result)
    plot_full_space_and_fc(result)
    plot_fc_translation_breakdown(result)
    plot_per_network(result)
    plot_tuning_surface(result)
    print(f"wrote plots to {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
