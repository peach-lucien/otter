"""Test 2b — Subtype CONTRAST translation through π.

Test 2 (script 04) had a confound: per-subtype absolute intensity per network is
dominated by network-size (large human networks like Subcortical collect most of
the predicted mass under almost any translation). The permuted-π null was non-zero
which means our test had no power to distinguish HOMER-specific signal from generic
column-sum effects.

The cleaner test: use the **subtype contrast** Δ = hyper − hypo. Network-size
appears equally in both subtypes and cancels out. The remaining signal is *pattern*:
which networks are differentially perturbed between hyper and hypo subtypes?

Procedure:
  1. mouse_delta[net] = mouse_intensity_hyper[net] − mouse_intensity_hypo[net]
     (9-vec, signed: positive → more perturbed in hyper; negative → more in hypo)
  2. human_delta_obs[net] = human_intensity_hyper[net] − human_intensity_hypo[net]
     (8-vec, signed)
  3. Distribute mouse_delta to mouse parcels → translate through π → aggregate to
     8 human networks → pred_human_delta (8-vec)
  4. Correlate pred_human_delta with human_delta_obs (Pearson + Spearman)
  5. Permuted-π null: shuffle π rows, repeat.

If pred_human_delta correlates with human_delta_obs significantly above null,
HOMER's π replicates the per-subtype spatial *contrast* — a real Pagani finding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
nc = import_module("01_network_crossvalidation")
st = import_module("04_subtype_translation")
assign_mouse_paper_networks = nc.assign_mouse_paper_networks
assign_human_paper_networks = nc.assign_human_paper_networks
load_pagani_subtype_matrices = st.load_pagani_subtype_matrices
network_intensity = st.network_intensity
mouse_intensity_to_parcel_values = st.mouse_intensity_to_parcel_values
aggregate_human_parcels_to_networks = st.aggregate_human_parcels_to_networks

from homer.data import load_cached


def main():
    print("=" * 80)
    print("Pagani 2026 Test 2b — subtype CONTRAST (hyper − hypo) translation through π")
    print("=" * 80)

    M, _ = load_cached("mouse", cache_dir="outputs/anndata")
    H, _ = load_cached("human", cache_dir="outputs/anndata")
    pi = np.load("outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")

    mouse_paper_net, mouse_net_names = assign_mouse_paper_networks(M.var, separate_aud=True)
    human_paper_net, human_net_names = assign_human_paper_networks(H.var, separate_aud=True)
    aud_idx = human_net_names.index("Auditory")
    som_idx = human_net_names.index("SomatoMotor")
    human_paper_net_merged = human_paper_net.copy()
    human_paper_net_merged[human_paper_net == aud_idx] = som_idx

    data = load_pagani_subtype_matrices()

    metric = "abs_rowcol_sum"
    mouse_int_hypo  = dict(zip(data["mouse_nets"], network_intensity(data["mouse_hypo"],  metric)))
    mouse_int_hyper = dict(zip(data["mouse_nets"], network_intensity(data["mouse_hyper"], metric)))
    human_int_hypo  = dict(zip(data["human_nets"], network_intensity(data["human_hypo"],  metric)))
    human_int_hyper = dict(zip(data["human_nets"], network_intensity(data["human_hyper"], metric)))

    # Subtype contrast vectors
    mouse_delta_dict = {k: mouse_int_hyper[k] - mouse_int_hypo[k]
                         for k in data["mouse_nets"]}
    human_delta_obs = np.array([human_int_hyper[n] - human_int_hypo[n]
                                 for n in data["human_nets"]])

    print(f"\nMouse Δ (hyper − hypo) per Pagani network:")
    for k, v in mouse_delta_dict.items():
        print(f"  {k:18s}: {v:+8.2f}")
    print(f"\nObserved human Δ (hyper − hypo) per Pagani network:")
    for n, v in zip(data["human_nets"], human_delta_obs):
        print(f"  {n:18s}: {v:+8.2f}")

    # Predict human Δ from mouse Δ via π
    mouse_delta_parcels = mouse_intensity_to_parcel_values(
        mouse_paper_net, mouse_net_names, mouse_delta_dict)
    pred_per_human_parcel = mouse_delta_parcels @ pi   # (2094,)
    pred_per_human_net = aggregate_human_parcels_to_networks(
        pred_per_human_parcel, human_paper_net_merged, human_net_names,
        target_names=data["human_nets"])
    pred_vec = np.array([pred_per_human_net[n] for n in data["human_nets"]])

    print(f"\nPredicted human Δ (via π) per Pagani network:")
    for n, v in zip(data["human_nets"], pred_vec):
        print(f"  {n:18s}: {v:+8.4f}")

    # Correlations
    r_p, p_p = pearsonr(pred_vec, human_delta_obs)
    r_s, p_s = spearmanr(pred_vec, human_delta_obs)
    print(f"\nPredicted-vs-observed subtype-contrast correlation (8 human nets):")
    print(f"  Pearson  r = {r_p:+.3f} (p = {p_p:.3f})")
    print(f"  Spearman ρ = {r_s:+.3f} (p = {p_s:.3f})")

    # Permuted-π null
    print(f"\nPermuted-π null (200 row-shuffles):")
    rng = np.random.default_rng(seed=42)
    n_trials = 200
    null_rs_p = []
    null_rs_s = []
    for _ in range(n_trials):
        perm = rng.permutation(pi.shape[0])
        pi_n = pi[perm]
        pred_n = aggregate_human_parcels_to_networks(
            mouse_delta_parcels @ pi_n, human_paper_net_merged, human_net_names,
            target_names=data["human_nets"])
        pn = np.array([pred_n[n] for n in data["human_nets"]])
        rp, _ = pearsonr(pn, human_delta_obs)
        rs, _ = spearmanr(pn, human_delta_obs)
        null_rs_p.append(rp); null_rs_s.append(rs)
    null_p_mean = float(np.mean(null_rs_p))
    null_p_ci = (float(np.percentile(null_rs_p, 2.5)),
                 float(np.percentile(null_rs_p, 97.5)))
    null_s_mean = float(np.mean(null_rs_s))
    null_s_ci = (float(np.percentile(null_rs_s, 2.5)),
                 float(np.percentile(null_rs_s, 97.5)))
    # one-sided p (fraction of null trials with r >= real r)
    p_one_sided_p = float(np.mean(np.array(null_rs_p) >= r_p))
    p_one_sided_s = float(np.mean(np.array(null_rs_s) >= r_s))
    print(f"  Pearson  null mean={null_p_mean:+.3f}, 95% CI {null_p_ci}, "
          f"  empirical one-sided p (real ≥ null): {p_one_sided_p:.3f}")
    print(f"  Spearman null mean={null_s_mean:+.3f}, 95% CI {null_s_ci}, "
          f"  empirical one-sided p (real ≥ null): {p_one_sided_s:.3f}")

    # Verdict
    print(f"\nVerdict:")
    if r_p > null_p_ci[1] and r_s > null_s_ci[1]:
        verdict = "STRONG: real correlation above 95% null CI on both metrics"
    elif r_p > null_p_ci[1] or r_s > null_s_ci[1]:
        verdict = "PARTIAL: real correlation above 95% null CI on one metric"
    elif r_p > null_p_mean and r_s > null_s_mean:
        verdict = "WEAK: real correlation above null mean but within 95% CI"
    else:
        verdict = "NULL: real correlation not distinguishable from null"
    print(f"  {verdict}")

    out = {
        "metric": metric,
        "pi_file": "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy",
        "mouse_delta":          mouse_delta_dict,
        "human_delta_observed": dict(zip(data["human_nets"], human_delta_obs.tolist())),
        "human_delta_predicted": dict(zip(data["human_nets"], pred_vec.tolist())),
        "pearson_r":  r_p, "pearson_p": p_p,
        "spearman_r": r_s, "spearman_p": p_s,
        "null": {
            "n_trials": n_trials,
            "pearson_mean":  null_p_mean,
            "pearson_ci95":  list(null_p_ci),
            "spearman_mean": null_s_mean,
            "spearman_ci95": list(null_s_ci),
            "pearson_one_sided_p":  p_one_sided_p,
            "spearman_one_sided_p": p_one_sided_s,
        },
        "verdict": verdict,
    }
    out_path = Path("outputs/logs/autism_subtypes_contrast.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
