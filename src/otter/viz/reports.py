"""Comprehensive comparison report builder.

Pulls all evaluation results from `outputs/logs/*.json` and produces:
    - wide DataFrame (configs × headline metrics)
    - long DataFrame (configs × networks → top1)
    - markdown summary
    - matplotlib figures (multi-metric bars + per-network heatmap)

Design: keep each step a pure function returning data. The pipeline orchestrator
(`pipeline/07_build_artefacts.py`) handles file I/O.

Public:
    build_comparison_table(logs_dir) -> (wide_df, long_df, null_z_dict, bootstrap_dict)
    aggregate_anchor_cv(per_network_dict) -> dict
    aggregate_null(null_per_network_dict, *, weights_per_net) -> dict
    render_summary_md(wide_df, long_df, null_z_dict, bootstrap_dict) -> str
    make_comparison_bars_figure(wide_df, *, headline_configs)
    make_per_network_heatmap_figure(long_df)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from otter.data.networks import NETWORKS


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def aggregate_anchor_cv(per_net: dict) -> dict:
    """Weighted-by-n_anchors_held aggregate across networks."""
    if not per_net or any("n_anchors_held" not in v for v in per_net.values()):
        return {}
    nets = [n for n in NETWORKS if n in per_net]
    if not nets:
        return {}
    w = np.array([per_net[n]["n_anchors_held"] for n in nets], dtype=float)
    wt = w.sum()
    out = {"n_networks": len(nets), "n_anchors_total": int(wt)}
    for k in ["top1", "top5", "pair_id", "hemi", "mean_rank", "mean_xyz_dist"]:
        vals = np.array([per_net[n].get(k, np.nan) for n in nets], dtype=float)
        if np.isnan(vals).all():
            out[f"weighted_{k}"] = float("nan")
        else:
            out[f"weighted_{k}"] = float(np.nansum(vals * w) / wt)
    return out


def aggregate_full_space(per_net: dict) -> dict:
    """Weighted (by # held-out anchors) aggregate of full-space metrics across
    networks. Mirrors aggregate_anchor_cv but for the full-space JSON shape."""
    if not per_net:
        return {}
    nets = [n for n in NETWORKS if n in per_net]
    if not nets:
        return {}
    w = np.array([per_net[n].get("n", 0) for n in nets], dtype=float)
    wt = w.sum()
    if wt == 0:
        return {}
    out = {"n_networks": len(nets), "n_anchors_total": int(wt)}
    for k in ("full_top1", "full_top5", "mean_rank_full",
              "frac_argmax_is_anchor", "frac_in_neighborhood",
              "mean_xyz_dist_full", "mean_mass_on_correct_anchor"):
        vals = np.array([per_net[n].get(k, np.nan) for n in nets], dtype=float)
        out[k] = float("nan") if np.isnan(vals).all() else float(np.nansum(vals * w) / wt)
    return out


def aggregate_null(null_per_net: dict, *, weights_per_net: dict,
                    key: str = "top1") -> dict:
    """Compute weighted-mean null distribution across networks.

    For each null trial t, combine per-network values into a weighted mean
    (weight = n_anchors_held / total). Then take mean and std across trials.
    """
    nets = [n for n in NETWORKS if n in null_per_net]
    if not nets:
        return {}
    w = np.array([weights_per_net[n] for n in nets], dtype=float)
    w = w / w.sum()
    n_trials_min = min(len(null_per_net[n]) for n in nets)
    if n_trials_min < 2:
        return {}
    vals = np.stack([
        [null_per_net[n][t][key] for n in nets]
        for t in range(n_trials_min)
    ])
    weighted_per_trial = vals @ w
    return {
        f"null_{key}_mean":     float(weighted_per_trial.mean()),
        f"null_{key}_std":      float(weighted_per_trial.std()),
        f"null_{key}_n_trials": int(n_trials_min),
    }


# ---------------------------------------------------------------------------
# Comparison table assembly
# ---------------------------------------------------------------------------
def _safe_load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# Each entry: name → (display_label, source_log, key_in_log, notes)
# source_log is the dict loaded from the JSON; "_self" means the entire log
# IS the per-network dict (used by hierarchical_cv.json which doesn't nest).
def _config_origin(mcv, hier, itr) -> dict:
    return {
        "baseline_fc_only":              ("baseline (FC only)",            mcv,  "baseline_fc_only",            ""),
        "fc_plus_xyz_gw":                ("FC + xyz GW",                    mcv,  "fc_plus_xyz_gw",              ""),
        "fc_plus_network_mask":          ("FC + network mask",              mcv,  "fc_plus_network_mask",        ""),
        "fc_plus_SC":                    ("FC + SC (production)",           mcv,  "fc_plus_SC",                  "production"),
        "fc_plus_gene_GW":               ("FC + gene GW",                   mcv,  "fc_plus_gene_GW",             ""),
        "fc_plus_M_gene":                ("FC + M_gene",                    mcv,  "fc_plus_M_gene",              ""),
        "fc_plus_SC_plus_M_gene":        ("FC + SC + M_gene",               mcv,  "fc_plus_SC_plus_M_gene",      ""),
        "all_modalities":                ("all modalities (FC+xyz+SC+gene)", mcv, "all_modalities",              ""),
        "fc_plus_selective_M_gene":      ("FC + selective M_gene",          mcv,  "fc_plus_selective_M_gene",    ""),
        "fc_plus_SC_plus_selective_M_gene": ("FC + SC + selective M_gene",  mcv,  "fc_plus_SC_plus_selective_M_gene", ""),
        "all_modalities_selective":      ("all modalities (selective M_gene)", mcv, "all_modalities_selective",   ""),
        "fc_plus_M_anchor":              ("FC + M_anchor (item A)",         mcv,  "fc_plus_M_anchor",            "negative"),
        "fc_plus_SC_plus_M_anchor":      ("FC + SC + M_anchor (item A)",    mcv,  "fc_plus_SC_plus_M_anchor",    "negative"),
        "hierarchical":                  ("Hierarchical (per-network)",     hier, "_self",                       "M4: cleaner WN, hurts CV"),
        "iterative_soft_lam0.30":        ("Iterative soft (lam=0.30, item B)", itr, "fc_plus_SC__iter2_topk200_thr0.50_lam0.30", "no-op"),
        "iterative_hard_lam1.00":        ("Iterative hard (lam=1.00, item B)", itr, "fc_plus_SC__iter2_topk200_thr0.95_lam1.00_hard", "no-op"),
    }


_FC_TRANSLATION_KEYS = {
    "baseline_fc_only":     "baseline_fc_only",
    "fc_plus_xyz_gw":       "fc_plus_xyz_gw",
    "fc_plus_network_mask": "fc_plus_network_mask",
    "fc_plus_SC":           "fc_plus_SC",
    "hierarchical":         "hierarchical_fc_only",
}
_SUBJECT_CV_KEYS = {
    "baseline_fc_only": "fc_only",
    "fc_plus_SC":       "fc_plus_SC",
}


def build_per_network_long(logs_dir: str | Path) -> pd.DataFrame:
    """Long-form DataFrame: configs × networks → top1."""
    logs_dir = Path(logs_dir)
    mcv  = _safe_load(logs_dir / "multimodal_cv.json")
    hier = _safe_load(logs_dir / "hierarchical_cv.json")
    itr  = _safe_load(logs_dir / "iterative_cv.json")
    origin = _config_origin(mcv, hier, itr)
    rows = []
    for cfg_key, (label, src, src_key, _notes) in origin.items():
        per_net = src.get(src_key) if src_key != "_self" else src
        if not per_net:
            continue
        for net in NETWORKS:
            if net in per_net and "top1" in per_net[net]:
                rows.append({
                    "config":   cfg_key,
                    "label":    label,
                    "network":  net,
                    "top1":     per_net[net]["top1"],
                    "n":        per_net[net].get("n_anchors_held", per_net[net].get("n", np.nan)),
                })
    return pd.DataFrame(rows)


def build_comparison_table(logs_dir: str | Path) -> tuple:
    """Build the wide comparison table + null z-scores + bootstrap dict.

    Returns
    -------
    (wide_df, long_df, null_z_dict, bootstrap_dict)
    """
    logs_dir = Path(logs_dir)
    mcv  = _safe_load(logs_dir / "multimodal_cv.json")
    hier = _safe_load(logs_dir / "hierarchical_cv.json")
    itr  = _safe_load(logs_dir / "iterative_cv.json")
    fct  = _safe_load(logs_dir / "fc_translation.json")
    nd   = _safe_load(logs_dir / "null_distributions.json")
    # Bootstrap: prefer the per-config file (post-fix), fall back to the legacy
    # FC-only one if the new file isn't there yet
    boot_sc  = _safe_load(logs_dir / "bootstrap_summary_fc_plus_SC.json")
    boot_old = _safe_load(logs_dir / "bootstrap_summary.json")
    boot = boot_sc if boot_sc else boot_old
    full_space = _safe_load(logs_dir / "full_space_eval.json")

    origin = _config_origin(mcv, hier, itr)
    subject_cv = (fct or {}).get("subject_cv", {})

    rows = []
    for cfg_key, (label, src, src_key, notes) in origin.items():
        per_net = src.get(src_key) if src_key != "_self" else src
        anc = aggregate_anchor_cv(per_net or {})
        ft_key = _FC_TRANSLATION_KEYS.get(cfg_key)
        ft = (fct or {}).get(ft_key, {}) if ft_key else {}
        scv_key = _SUBJECT_CV_KEYS.get(cfg_key)
        scv = subject_cv.get(scv_key, {}) if scv_key else {}
        # NEW, full-space metrics from outputs/logs/full_space_eval.json
        # Aggregate weighted across networks if data exists for this config.
        fs_per_net = (full_space or {}).get(cfg_key, {})
        fs = aggregate_full_space(fs_per_net) if fs_per_net else {}

        rows.append({
            "config":                    cfg_key,
            "label":                     label,
            "notes":                     notes,
            # === Restricted-anchor CV (the "ranking" interpretation) ===
            "anchor_top1":               anc.get("weighted_top1", float("nan")),
            "anchor_top5":               anc.get("weighted_top5", float("nan")),
            "anchor_pair":               anc.get("weighted_pair_id", float("nan")),
            "anchor_hemi":               anc.get("weighted_hemi", float("nan")),
            "anchor_mean_rank":          anc.get("weighted_mean_rank", float("nan")),
            "anchor_mean_xyz_dist":      anc.get("weighted_mean_xyz_dist", float("nan")),
            "n_anchors_total":           anc.get("n_anchors_total", 0),
            # === Full-space recovery (the "per-voxel mapping" interpretation) ===
            "full_top1":                 fs.get("full_top1", float("nan")),
            "full_top5":                 fs.get("full_top5", float("nan")),
            "full_mean_rank":            fs.get("mean_rank_full", float("nan")),
            "full_argmax_is_anchor":     fs.get("frac_argmax_is_anchor", float("nan")),
            "full_in_neighborhood":      fs.get("frac_in_neighborhood", float("nan")),
            "full_mass_on_correct":      fs.get("mean_mass_on_correct_anchor", float("nan")),
            # === FC translation ===
            "fc_translation_r":          ft.get("pearson_r_overall", float("nan")),
            "fc_translation_within_net": ft.get("pearson_r_within_net", float("nan")),
            "fc_translation_cross_net":  ft.get("pearson_r_cross_net", float("nan")),
            "fc_translation_n_kept":     ft.get("n_human_nodes_kept", 0),
            # === Subject CV (held-out) ===
            "subject_cv_train_r":        scv.get("train_r_mean", float("nan")),
            "subject_cv_test_r":         scv.get("test_r_mean", float("nan")),
            "subject_cv_test_r_std":     scv.get("test_r_std", float("nan")),
            "subject_cv_gap":            scv.get("gap_mean", float("nan")),
        })

    wide_df = pd.DataFrame(rows)

    # Long form for the heatmap
    long_df = build_per_network_long(logs_dir)

    # Null z-scores vs production
    null_z = {}
    if nd and mcv.get("fc_plus_SC"):
        prod = mcv["fc_plus_SC"]
        weights = {n: prod[n]["n_anchors_held"] for n in NETWORKS if n in prod}
        for null_kind in ("random_pi", "permuted_anchors"):
            agg = aggregate_null(nd.get(null_kind, {}),
                                  weights_per_net=weights, key="top1")
            if not agg:
                continue
            real_top1 = aggregate_anchor_cv(prod)["weighted_top1"]
            null_mean = agg["null_top1_mean"]
            null_std  = agg["null_top1_std"]
            z = (real_top1 - null_mean) / max(null_std, 1e-9)
            null_z[null_kind] = {
                "real_top1": real_top1,
                "null_mean": null_mean,
                "null_std":  null_std,
                "n_trials":  agg.get("null_top1_n_trials", 0),
                "z_score":   z,
            }

    return wide_df, long_df, null_z, (boot or {})


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def render_summary_md(wide_df: pd.DataFrame, long_df: pd.DataFrame,
                       null_z: dict, bootstrap: dict) -> str:
    """Render the markdown summary as a single string."""
    md = []
    md.append("# Comprehensive comparison, all configs, all metrics\n")
    md.append(
        f"Generated from results in `outputs/logs/` on data of "
        f"{bootstrap.get('n_iterations', '?')}-iter bootstrap and 11-network "
        f"leave-one-network-out CV. Production config marked **bold**.\n"
    )
    md.append("## Headline table, restricted-anchor CV (the 'ranking' metric)\n")
    md.append("Top-1 here is **argmax restricted to the held-out anchor candidates**, "
              "NOT global argmax over all 2094 human nodes (which is in the next table).\n")
    md.append("FC-translation r is **in-sample** (uses the same fc_mean to build C_h that it evaluates).\n")
    md.append("| Config | rTop-1 | rTop-5 | Pair | Hemi | Rank | xyz_d | "
              "FC-r overall | FC-r within | FC-r cross | Subj-CV test r | Notes |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    def _fmt(x, fs="%.2f"):
        try: return ", " if not np.isfinite(x) else fs % x
        except Exception: return ", "

    def _pct(x):
        try: return ", " if not np.isfinite(x) else f"{x:.0%}"
        except Exception: return ", "

    for _, r in wide_df.iterrows():
        if (not np.isfinite(r["anchor_top1"])
            and not np.isfinite(r["fc_translation_r"])
            and not np.isfinite(r["subject_cv_test_r"])):
            continue
        nl = f"**{r['label']}**" if r["notes"] == "production" else r["label"]
        notes_extra = r.get("notes", "") or ""
        n_nets_run = (long_df[long_df["config"] == r["config"]]).shape[0]
        if 0 < n_nets_run < 11:
            notes_extra = (notes_extra + f"; {n_nets_run}/11 nets only").lstrip("; ")
        md.append(
            f"| {nl} | {_pct(r['anchor_top1'])} | {_pct(r['anchor_top5'])} | "
            f"{_pct(r['anchor_pair'])} | {_pct(r['anchor_hemi'])} | "
            f"{_fmt(r['anchor_mean_rank'])} | {_fmt(r['anchor_mean_xyz_dist'], '%.3f')} | "
            f"{_fmt(r['fc_translation_r'])} | {_fmt(r['fc_translation_within_net'])} | "
            f"{_fmt(r['fc_translation_cross_net'])} | {_fmt(r['subject_cv_test_r'])} | "
            f"{notes_extra} |"
        )
    md.append("")

    # Only show the full-space table for configs that have data
    has_fs = any(np.isfinite(r) for r in wide_df["full_top1"].values)
    if has_fs:
        md.append("## Full-space recovery, global argmax over all 2094 human nodes\n")
        md.append(
            "The conservative per-voxel metric. The model's full-space argmax typically lands "
            "on a non-anchor *grid* node near the correct anchor rather than the anchor "
            "itself. Full-top-1 is much smaller than restricted-top-1 because the search "
            "space is 2094× larger.\n"
        )
        md.append("| Config | full-Top-1 | full-Top-5 | mean rank /2094 | argmax is anchor | "
                  "in 5% neighborhood | mean mass on correct anchor |")
        md.append("|---|---|---|---|---|---|---|")
        for _, r in wide_df.iterrows():
            if not np.isfinite(r["full_top1"]):
                continue
            nl = f"**{r['label']}**" if r["notes"] == "production" else r["label"]
            md.append(
                f"| {nl} | {r['full_top1']:.0%} | {r['full_top5']:.0%} | "
                f"{r['full_mean_rank']:.0f} | {r['full_argmax_is_anchor']:.0%} | "
                f"{r['full_in_neighborhood']:.0%} | {r['full_mass_on_correct']:.3f} |"
            )
        md.append("")

    md.append("## Null calibration (production = `fc_plus_SC`)\n")
    md.append("Each cell of the null is a per-trial weighted-mean top-1 across all 11 networks.\n")
    md.append("| Null kind | n trials | Real top-1 | Null mean | Null std | z-score |")
    md.append("|---|---|---|---|---|---|")
    for k, v in null_z.items():
        md.append(
            f"| {k} | {v['n_trials']} | {v['real_top1']:.0%} | "
            f"{v['null_mean']:.0%} | {v['null_std']:.0%} | {v['z_score']:+.1f} |"
        )
    md.append("")

    if bootstrap:
        md.append("## Bootstrap stability (production solve)\n")
        md.append(f"- iterations: {bootstrap.get('n_iterations', '?')}")
        md.append(f"- mean per-cell stability: {bootstrap.get('mean_stability', 0):.3f}")
        md.append(f"- median: {bootstrap.get('median_stability', 0):.3f}")
        md.append(f"- frac stable above 0.8: {bootstrap.get('frac_stable_above_0.8', 0):.1%}")
        md.append(f"- frac stable above 0.5: {bootstrap.get('frac_stable_above_0.5', 0):.1%}\n")

    md.append("## Per-network top-1 heatmap (see fig 14)\n")
    md.append(
        "Variance across networks is the story, most configs land 100% on the easy "
        "networks (auditory, frontoparietal, frontal_dmn, etc.) and 25-50% on visual / "
        "brainstem / sensorimotor.\n"
    )
    return "\n".join(md)


# ---------------------------------------------------------------------------
# Matplotlib figures
# ---------------------------------------------------------------------------
def make_comparison_bars_figure(wide_df: pd.DataFrame, *,
                                 headline_configs: Optional[list[str]] = None):
    """4-panel matplotlib figure: anchor_top1, anchor_top5, mean_xyz_dist,
    fc_translation_r. Returns the Figure object (caller saves)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headline_configs = headline_configs or [
        "baseline_fc_only", "fc_plus_xyz_gw", "fc_plus_network_mask",
        "fc_plus_SC", "fc_plus_M_gene", "fc_plus_M_anchor",
        "fc_plus_SC_plus_M_gene", "all_modalities", "hierarchical",
    ]
    sub = wide_df.set_index("config").loc[
        [h for h in headline_configs if h in wide_df["config"].values]
    ]
    metrics = [
        ("anchor_top1",          "Anchor top-1"),
        ("anchor_top5",          "Anchor top-5"),
        ("anchor_mean_xyz_dist", "Mean xyz distance (lower=better)"),
        ("fc_translation_r",     "FC translation r"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        vals = sub[col].values
        labels = sub["label"].values
        bars = ax.barh(range(len(sub)), vals,
                        color=["#4c72b0" if c != "fc_plus_SC" else "#dd8452"
                                for c in sub.index])
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(title)
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                fmt = "%.0f%%" if "top" in col else "%.3f"
                disp = (v * 100) if "top" in col else v
                ax.text(b.get_width() * 1.02 if v > 0 else 0.01,
                         b.get_y() + b.get_height() / 2,
                         fmt % disp, va="center", fontsize=8)
    fig.suptitle("Comprehensive comparison, all configs (production = orange)",
                  fontsize=13)
    fig.tight_layout()
    return fig


def make_per_network_heatmap_figure(long_df: pd.DataFrame):
    """Configs × networks heatmap (matplotlib). Returns the Figure object."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = long_df.pivot_table(index="label", columns="network",
                                  values="top1", aggfunc="first")
    pivot = pivot[[n for n in NETWORKS if n in pivot.columns]]
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                         color="white" if v < 0.4 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Held-out anchor top-1 accuracy")
    ax.set_title("Configs × networks: held-out anchor top-1 (leave-one-network-out CV)",
                  fontsize=12)
    fig.tight_layout()
    return fig
