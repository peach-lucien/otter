"""Rerun Test 3 (gene-spatial translation) using the expanded Pagani gene matrix.

Prerequisite: download_pagani_ish.py must have been run first, producing
  - pagani_mouse_expr.npy            (1864, n_genes_kept)
  - pagani_gene_list_resolved.csv

Output:
  - outputs/logs/autism_subtypes_gene_spatial_expanded.json
  - console: same diagnostics as 09_gene_spatial_translation.py but with full
    gene coverage
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parents[1]))   # experiments/autism_subtypes/
from importlib import import_module
nc = import_module("01_network_crossvalidation")
st = import_module("04_subtype_translation")
assign_human_paper_networks = nc.assign_human_paper_networks
load_pagani_subtype_matrices = st.load_pagani_subtype_matrices
network_intensity = st.network_intensity

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached


def main():
    print("=" * 80)
    print("Pagani 2026 Test 3 — gene-spatial translation, EXPANDED gene panel")
    print("=" * 80)

    expr_path = Path(__file__).parent / "pagani_mouse_expr.npy"
    meta_path = Path(__file__).parent / "pagani_gene_list_resolved.csv"
    if not expr_path.exists():
        print(f"\nERROR: {expr_path} not found.")
        print(f"Run download_pagani_ish.py first.")
        sys.exit(1)

    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs" / "anndata"))
    pi = np.load(str(ROOT / "outputs" / "coupling" / "pi_fc_plus_SC_with_all_packs.npy"))

    expr = np.load(expr_path)
    meta = pd.read_csv(meta_path)
    print(f"\nExpanded gene matrix: {expr.shape}")
    print(f"  subtypes: hypo={(meta['subtype']=='hypo').sum()}, "
          f"hyper={(meta['subtype']=='hyper').sum()}")

    # NaN handling
    expr = expr.copy()
    col_mean = np.nanmean(expr, axis=0)
    inds = np.where(np.isnan(expr))
    expr[inds] = np.take(col_mean, inds[1])

    # z-score per gene
    z = (expr - expr.mean(0, keepdims=True)) / (expr.std(0, keepdims=True) + 1e-9)

    hypo_idx = np.where(meta["subtype"] == "hypo")[0]
    hyper_idx = np.where(meta["subtype"] == "hyper")[0]

    # Per-parcel mouse spatial score for each subtype
    hypo_score  = z[:, hypo_idx].mean(axis=1)
    hyper_score = z[:, hyper_idx].mean(axis=1)
    print(f"\nMouse-side z-scored scores:")
    print(f"  hypo:  range {hypo_score.min():+.3f} .. {hypo_score.max():+.3f}, n_genes={len(hypo_idx)}")
    print(f"  hyper: range {hyper_score.min():+.3f} .. {hyper_score.max():+.3f}, n_genes={len(hyper_idx)}")

    # Translate via π
    hypo_pred_h  = hypo_score  @ pi
    hyper_pred_h = hyper_score @ pi

    # Aggregate to 8 Pagani human networks
    human_net, human_paper_names = assign_human_paper_networks(H.var, separate_aud=True)
    aud_idx = human_paper_names.index("Auditory")
    som_idx = human_paper_names.index("SomatoMotor")
    human_net = human_net.copy()
    human_net[human_net == aud_idx] = som_idx
    pagani_human = ["Control", "DMN", "DorsAtten", "Limbic",
                    "Salience", "SomatoMotor", "Visual", "Subcortical"]
    h_name_to_idx = {n: human_paper_names.index(n) for n in pagani_human}
    pag_net = np.full_like(human_net, -1)
    for new_i, n in enumerate(pagani_human):
        pag_net[human_net == h_name_to_idx[n]] = new_i

    def agg(values):
        out = np.zeros(len(pagani_human))
        for i in range(len(pagani_human)):
            m = pag_net == i
            if m.any():
                out[i] = values[m].mean()
        return out

    pred_delta = agg(hyper_pred_h) - agg(hypo_pred_h)

    # Observed Δ from Fig 4e
    data = load_pagani_subtype_matrices()
    obs_hypo = network_intensity(data["human_hypo"], "abs_rowcol_sum")
    obs_hyper = network_intensity(data["human_hyper"], "abs_rowcol_sum")
    obs_delta = obs_hyper - obs_hypo

    # Correlations
    r_p, p_p = pearsonr(pred_delta, obs_delta)
    r_s, p_s = spearmanr(pred_delta, obs_delta)
    same_sign = int((np.sign(pred_delta) == np.sign(obs_delta)).sum())
    print(f"\nPer-network Δ (hyper − hypo):")
    print(f"  {'net':<18s} | {'obs':>10s} | {'pred':>12s}")
    for i, n in enumerate(pagani_human):
        print(f"  {n:<18s} | {obs_delta[i]:>+10.2f} | {pred_delta[i]:>+12.4f}")
    print(f"\nPearson r = {r_p:+.3f} (p = {p_p:.3f})")
    print(f"Spearman ρ = {r_s:+.3f} (p = {p_s:.3f})")
    print(f"Same-sign per network: {same_sign}/8")

    # Permuted-π null
    print(f"\nPermuted-π null (200 trials):")
    rng = np.random.default_rng(seed=42)
    null_p, null_s = [], []
    for _ in range(200):
        perm = rng.permutation(pi.shape[0])
        pi_n = pi[perm]
        pn = agg(hyper_score @ pi_n) - agg(hypo_score @ pi_n)
        rp, _ = pearsonr(pn, obs_delta)
        rs, _ = spearmanr(pn, obs_delta)
        null_p.append(rp); null_s.append(rs)
    null_p, null_s = np.array(null_p), np.array(null_s)
    print(f"  Pearson  null mean={null_p.mean():+.3f}, "
          f"95% CI ({np.percentile(null_p, 2.5):+.3f}, {np.percentile(null_p, 97.5):+.3f}), "
          f"empirical p = {(null_p >= r_p).mean():.3f}")
    print(f"  Spearman null mean={null_s.mean():+.3f}, "
          f"95% CI ({np.percentile(null_s, 2.5):+.3f}, {np.percentile(null_s, 97.5):+.3f}), "
          f"empirical p = {(null_s >= r_s).mean():.3f}")

    out = {
        "n_genes_total": int(expr.shape[1]),
        "n_hypo":  int(len(hypo_idx)),
        "n_hyper": int(len(hyper_idx)),
        "pearson_r": float(r_p), "pearson_p_analytical": float(p_p),
        "spearman_r": float(r_s), "spearman_p_analytical": float(p_s),
        "pearson_empirical_p":  float((null_p >= r_p).mean()),
        "spearman_empirical_p": float((null_s >= r_s).mean()),
        "same_sign_count": same_sign,
        "human_networks": pagani_human,
        "observed_delta":  obs_delta.tolist(),
        "predicted_delta": pred_delta.tolist(),
    }
    out_path = ROOT / "outputs" / "logs" / "autism_subtypes_gene_spatial_expanded.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
