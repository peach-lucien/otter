"""Test 2 — Subtype spatial-pattern translation through HOMER's π.

Pagani et al. 2026 report per-subtype network connectivity matrices in both
species (ED Fig 1 for mouse, Fig 4e for human). Each subtype (hyper, hypo) has
a spatial signature — which networks carry the strongest connectivity
perturbation in that subtype.

Their claim: "the FC subtypes recur cross-species in matching anatomical
locations" (their claim 3 — supported by **name-based** matching of mouse
Somatomotor to human Somatomotor, etc.).

This test replaces the name-based bridge with HOMER's quantitative π:
  1. Read the mouse 9-network per-subtype intensity vector from ED Fig 1.
  2. Distribute the per-network intensity to 1864 mouse parcels (via
     PAIRID_TO_NETWORK → paper-mouse-network labels).
  3. Translate through π: pred[h] = Σ_m π[m, h] · mouse_intensity[m]
  4. Aggregate predicted per-parcel values to 8 human networks via Schaefer-Yeo7.
  5. Compare to the observed human 8-network intensity vector from Fig 4e
     using Pearson correlation.
  6. **Subtype-specificity check**: corr(pred_hypo, obs_hypo) should exceed
     corr(pred_hypo, obs_hyper) — if π is informative, the mouse hypo
     spatial pattern should predict the *human hypo* pattern better than
     the *human hyper* pattern.

Tests one of Pagani's actual claims (subtype spatial patterns recur in
matching locations), not their scaffolding.

Usage:
    PYTHONPATH=src python experiments/autism_subtypes/04_subtype_translation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import openpyxl
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
mod = import_module("01_network_crossvalidation")
assign_mouse_paper_networks = mod.assign_mouse_paper_networks
assign_human_paper_networks = mod.assign_human_paper_networks

from homer.data import load_cached


import os
# Sandbox path → user-machine fallback. Override via PAGANI_XLSX env var.
_DEFAULT_SANDBOX = "/sessions/wizardly-admiring-tesla/mnt/uploads/41593_2026_2287_MOESM6_ESM.xlsx"
_USER_FALLBACKS = [
    Path(__file__).resolve().parents[2] / "data_external" / "pagani_2026" / "41593_2026_2287_MOESM6_ESM.xlsx",
    Path(__file__).resolve().parent / "41593_2026_2287_MOESM6_ESM.xlsx",
    Path(__file__).resolve().parents[2] / "41593_2026_2287_MOESM6_ESM.xlsx",
]
PAGANI_XLSX = os.environ.get("PAGANI_XLSX")
if not PAGANI_XLSX:
    if Path(_DEFAULT_SANDBOX).exists():
        PAGANI_XLSX = _DEFAULT_SANDBOX
    else:
        for p in _USER_FALLBACKS:
            if p.exists():
                PAGANI_XLSX = str(p); break
        if not PAGANI_XLSX:
            PAGANI_XLSX = _DEFAULT_SANDBOX  # final fallback; will error clearly if missing

# Mouse 9-network names (from ED Fig 1 of paper)
PAGANI_MOUSE_NETS = [
    "Auditory", "BF", "Caudate Putamen", "DMN", "HC",
    "Salience", "Somatomotor", "Thalamus", "Visual",
]
# Human 8-network names (from Fig 4e)
PAGANI_HUMAN_NETS = [
    "Control", "DMN", "DorsAtten", "Limbic",
    "Salience", "SomatoMotor", "Visual", "Subcortical",
]

# How to map HOMER's mouse-side networks (PAIRID_TO_NETWORK + assign_mouse_paper_networks
# names) onto Pagani's 9 mouse network names.
# Note our mouse_paper_networks gives 13 labels:
#   Visual, Auditory, SomatoMotor, DorsAtten, Salience, Limbic, Control, DMN,
#   Subcortical, HC_Limbic, BF_Olfactory, Frontoparietal, Brainstem
HOMER_MOUSENET_TO_PAGANI_MOUSE: dict[str, str] = {
    "Auditory":      "Auditory",
    "SomatoMotor":   "Somatomotor",
    "Visual":        "Visual",
    "Salience":      "Salience",
    "DMN":           "DMN",
    "HC_Limbic":     "HC",
    "BF_Olfactory":  "BF",
    "Subcortical":   "Thalamus",          # HOMER subcortical (pids 13/14/15/18/19) → split below
    "Frontoparietal": None,               # not in Pagani 9-net — drop
    "Brainstem":     None,                # not in Pagani 9-net — drop
    "Limbic":        None,                # never populated in our scheme (legacy slot)
    "Control":       None,
    "DorsAtten":     None,
}


def _read_subtype_matrix(ws, header_row: int, data_start_row: int, n_nets: int,
                          col_start: int = 2) -> np.ndarray:
    """Read an n × n matrix block."""
    M = np.zeros((n_nets, n_nets), dtype=float)
    for i in range(n_nets):
        for j in range(n_nets):
            v = ws.cell(data_start_row + i, col_start + j).value
            if v is None:
                v = 0.0
            M[i, j] = float(v)
    return M


def load_pagani_subtype_matrices() -> dict:
    wb = openpyxl.load_workbook(PAGANI_XLSX, data_only=True)

    # ED Fig 1 — mouse, 9 networks
    ws = wb["ED - Figure 1"]
    # Hypo block: header row 2, data rows 3..11 (9 nets), columns B..J (2..10)
    mouse_hypo = _read_subtype_matrix(ws, header_row=2, data_start_row=3,
                                       n_nets=9, col_start=2)
    # Hyper block: 'Hyperconnectivity' at row 14, header at row 15, data rows 16..24
    mouse_hyper = _read_subtype_matrix(ws, header_row=15, data_start_row=16,
                                        n_nets=9, col_start=2)

    # Fig 4e — human, 8 networks
    ws = wb["Figure 4e"]
    # Hypo block: header row 3, data rows 4..11, cols B..I (2..9)
    human_hypo = _read_subtype_matrix(ws, header_row=3, data_start_row=4,
                                       n_nets=8, col_start=2)
    # Hyper block: header row 3, data rows 4..11, cols N..U (14..21)
    human_hyper = _read_subtype_matrix(ws, header_row=3, data_start_row=4,
                                        n_nets=8, col_start=14)

    return {
        "mouse_hypo":  mouse_hypo,
        "mouse_hyper": mouse_hyper,
        "human_hypo":  human_hypo,
        "human_hyper": human_hyper,
        "mouse_nets":  PAGANI_MOUSE_NETS,
        "human_nets":  PAGANI_HUMAN_NETS,
    }


def network_intensity(M: np.ndarray, metric: str = "abs_rowcol_sum") -> np.ndarray:
    """Reduce an n × n subtype matrix to a per-network intensity vector.

    metric:
      - "rowcol_sum": row + column sum (symmetrize then sum)
      - "abs_rowcol_sum": |M|.sum(axis=0) + |M|.sum(axis=1) — magnitude only
      - "rms": sqrt(mean(M**2)) per network row
    """
    if metric == "rowcol_sum":
        return M.sum(axis=1) + M.sum(axis=0) - np.diag(M)
    if metric == "abs_rowcol_sum":
        Ma = np.abs(M)
        return Ma.sum(axis=1) + Ma.sum(axis=0) - np.diag(Ma)
    if metric == "rms":
        return np.sqrt((M ** 2).mean(axis=1))
    raise ValueError(metric)


def mouse_intensity_to_parcel_values(mouse_paper_net: np.ndarray,
                                      mouse_net_names: list[str],
                                      mouse_net_intensity: dict[str, float],
                                      ) -> np.ndarray:
    """Distribute a per-network intensity dict to 1864 mouse parcels.
    Parcels in networks not in mouse_net_intensity get value 0.
    """
    n = len(mouse_paper_net)
    out = np.zeros(n, dtype=float)
    for net_idx, name in enumerate(mouse_net_names):
        pagani_name = HOMER_MOUSENET_TO_PAGANI_MOUSE.get(name)
        if pagani_name is None:
            continue
        val = mouse_net_intensity.get(pagani_name, 0.0)
        out[mouse_paper_net == net_idx] = val
    return out


def aggregate_human_parcels_to_networks(values: np.ndarray,
                                         human_paper_net: np.ndarray,
                                         human_net_names: list[str],
                                         target_names: list[str],
                                         ) -> dict[str, float]:
    """Aggregate per-human-parcel predicted values to per-network means."""
    out: dict[str, float] = {}
    for tname in target_names:
        # Map paper-network name → our internal name; "Subcortical" matches directly.
        # Our human_paper_networks uses ['Visual','Auditory','SomatoMotor','DorsAtten',
        #   'Salience','Limbic','Control','DMN','Subcortical']
        # If target is "DorsAtten" — same. "SomatoMotor" — same (modulo case). etc.
        if tname not in human_net_names:
            out[tname] = float("nan")
            continue
        idx = human_net_names.index(tname)
        mask = human_paper_net == idx
        out[tname] = float(values[mask].mean()) if mask.any() else 0.0
    return out


def main():
    print("=" * 80)
    print("Pagani 2026 Test 2 — subtype spatial-pattern translation through π")
    print("=" * 80)

    M, _ = load_cached("mouse", cache_dir="outputs/anndata")
    H, _ = load_cached("human", cache_dir="outputs/anndata")
    pi = np.load("outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")
    print(f"π: {pi.shape}, total mass: {pi.sum():.4f}")

    # Per-parcel network assignments (separate_aud=True gives 9 human nets here
    # but the paper only names 8 — we'll merge Auditory into SomatoMotor for the
    # human comparison since Pagani's "SomatoMotor" Yeo-7 collapses auditory back in).
    mouse_paper_net, mouse_net_names = assign_mouse_paper_networks(M.var, separate_aud=True)
    human_paper_net, human_net_names = assign_human_paper_networks(H.var, separate_aud=True)

    # Merge our human "Auditory" back into SomatoMotor for Pagani-comparison purposes
    aud_idx = human_net_names.index("Auditory")
    som_idx = human_net_names.index("SomatoMotor")
    human_paper_net_merged = human_paper_net.copy()
    human_paper_net_merged[human_paper_net == aud_idx] = som_idx
    # We don't shrink the name list because aggregate_human_parcels_to_networks
    # just looks up by name.

    data = load_pagani_subtype_matrices()

    # Use absolute row+col sum as intensity metric
    metric = "abs_rowcol_sum"
    print(f"\nNetwork-intensity metric: {metric}\n")

    results = {}
    for subtype in ["hypo", "hyper"]:
        print(f"--- Subtype: {subtype} ---")
        mouse_M = data[f"mouse_{subtype}"]
        human_M = data[f"human_{subtype}"]

        mouse_intensity = network_intensity(mouse_M, metric=metric)
        human_intensity_obs = network_intensity(human_M, metric=metric)
        mouse_int_dict = dict(zip(data["mouse_nets"], mouse_intensity))
        human_int_dict_obs = dict(zip(data["human_nets"], human_intensity_obs))

        print(f"  mouse-side intensity per Pagani net: "
              f"{ {k: round(v,2) for k,v in mouse_int_dict.items()} }")

        # Distribute to mouse parcels, translate, aggregate to human nets
        mouse_parcel_values = mouse_intensity_to_parcel_values(
            mouse_paper_net, mouse_net_names, mouse_int_dict)
        pred_per_human_parcel = mouse_parcel_values @ pi   # (2094,)
        pred_per_human_net = aggregate_human_parcels_to_networks(
            pred_per_human_parcel, human_paper_net_merged, human_net_names,
            target_names=data["human_nets"])

        print(f"  observed human intensity (Fig 4e):    "
              f"{ {k: round(v,2) for k,v in human_int_dict_obs.items()} }")
        print(f"  predicted human intensity (via π):    "
              f"{ {k: round(v,2) for k,v in pred_per_human_net.items()} }")

        results[subtype] = {
            "mouse_intensity": mouse_int_dict,
            "observed_human_intensity": human_int_dict_obs,
            "predicted_human_intensity": pred_per_human_net,
        }

    # Build aligned vectors for correlation
    target_human_nets = data["human_nets"]
    obs_hypo = np.array([results["hypo"]["observed_human_intensity"][n]
                         for n in target_human_nets])
    obs_hyper = np.array([results["hyper"]["observed_human_intensity"][n]
                          for n in target_human_nets])
    pred_hypo = np.array([results["hypo"]["predicted_human_intensity"][n]
                          for n in target_human_nets])
    pred_hyper = np.array([results["hyper"]["predicted_human_intensity"][n]
                           for n in target_human_nets])

    # Standardize per-vector to remove scale effects (Pearson does this anyway)
    def corr_pair(a, b):
        r, p = pearsonr(a, b)
        rs, ps = spearmanr(a, b)
        return {"pearson_r": float(r), "pearson_p": float(p),
                "spearman_r": float(rs), "spearman_p": float(ps)}

    print("\n" + "=" * 80)
    print("Correlations between predicted and observed (over 8 human networks)")
    print("=" * 80)
    matrix = {
        "pred_hypo_vs_obs_hypo":   corr_pair(pred_hypo, obs_hypo),
        "pred_hypo_vs_obs_hyper":  corr_pair(pred_hypo, obs_hyper),
        "pred_hyper_vs_obs_hyper": corr_pair(pred_hyper, obs_hyper),
        "pred_hyper_vs_obs_hypo":  corr_pair(pred_hyper, obs_hypo),
    }
    for k, v in matrix.items():
        print(f"  {k:30s} Pearson r={v['pearson_r']:+.3f} (p={v['pearson_p']:.3f}), "
              f"Spearman={v['spearman_r']:+.3f}")

    # Subtype-specificity check
    print("\nSubtype-specificity:")
    hypo_specific = matrix["pred_hypo_vs_obs_hypo"]["pearson_r"] > matrix["pred_hypo_vs_obs_hyper"]["pearson_r"]
    hyper_specific = matrix["pred_hyper_vs_obs_hyper"]["pearson_r"] > matrix["pred_hyper_vs_obs_hypo"]["pearson_r"]
    print(f"  predicted-hypo agrees with observed-hypo more than with observed-hyper? {hypo_specific}")
    print(f"  predicted-hyper agrees with observed-hyper more than with observed-hypo? {hyper_specific}")

    # Permuted-π null
    print("\nPermuted-π null (50 row-shuffles):")
    rng = np.random.default_rng(seed=42)
    null_rs = {"pred_hypo_vs_obs_hypo": [], "pred_hyper_vs_obs_hyper": []}
    null_specific_hypo = 0
    null_specific_hyper = 0
    n_trials = 50
    for trial in range(n_trials):
        perm = rng.permutation(pi.shape[0])
        pi_n = pi[perm]
        # Mouse intensity → parcels via SAME mouse_paper_net (we shuffle π, not mouse labels)
        mouse_pv_h = mouse_intensity_to_parcel_values(
            mouse_paper_net, mouse_net_names,
            results["hypo"]["mouse_intensity"])
        mouse_pv_hy = mouse_intensity_to_parcel_values(
            mouse_paper_net, mouse_net_names,
            results["hyper"]["mouse_intensity"])
        pred_h = aggregate_human_parcels_to_networks(
            mouse_pv_h @ pi_n, human_paper_net_merged, human_net_names,
            target_names=data["human_nets"])
        pred_hy = aggregate_human_parcels_to_networks(
            mouse_pv_hy @ pi_n, human_paper_net_merged, human_net_names,
            target_names=data["human_nets"])
        ph_v = np.array([pred_h[n] for n in target_human_nets])
        phy_v = np.array([pred_hy[n] for n in target_human_nets])
        r_hh, _ = pearsonr(ph_v, obs_hypo)
        r_hhy, _ = pearsonr(ph_v, obs_hyper)
        r_hyhy, _ = pearsonr(phy_v, obs_hyper)
        r_hyh, _ = pearsonr(phy_v, obs_hypo)
        null_rs["pred_hypo_vs_obs_hypo"].append(r_hh)
        null_rs["pred_hyper_vs_obs_hyper"].append(r_hyhy)
        null_specific_hypo  += int(r_hh  > r_hhy)
        null_specific_hyper += int(r_hyhy > r_hyh)
    print(f"  null mean r(pred_hypo, obs_hypo):     "
          f"{np.mean(null_rs['pred_hypo_vs_obs_hypo']):+.3f} "
          f"(95% CI: {np.percentile(null_rs['pred_hypo_vs_obs_hypo'],2.5):+.3f}..."
          f"{np.percentile(null_rs['pred_hypo_vs_obs_hypo'],97.5):+.3f})")
    print(f"  null mean r(pred_hyper, obs_hyper):   "
          f"{np.mean(null_rs['pred_hyper_vs_obs_hyper']):+.3f} "
          f"(95% CI: {np.percentile(null_rs['pred_hyper_vs_obs_hyper'],2.5):+.3f}..."
          f"{np.percentile(null_rs['pred_hyper_vs_obs_hyper'],97.5):+.3f})")
    print(f"  null subtype-specific fraction (hypo):  {null_specific_hypo}/{n_trials}")
    print(f"  null subtype-specific fraction (hyper): {null_specific_hyper}/{n_trials}")

    out = {
        "metric": metric,
        "pi_file": "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy",
        "observed_human_hypo":   dict(zip(target_human_nets, obs_hypo.tolist())),
        "observed_human_hyper":  dict(zip(target_human_nets, obs_hyper.tolist())),
        "predicted_human_hypo":  dict(zip(target_human_nets, pred_hypo.tolist())),
        "predicted_human_hyper": dict(zip(target_human_nets, pred_hyper.tolist())),
        "correlations": matrix,
        "subtype_specific": {
            "hypo":  bool(hypo_specific),
            "hyper": bool(hyper_specific),
        },
        "null": {
            "pred_hypo_vs_obs_hypo": {
                "mean_r": float(np.mean(null_rs["pred_hypo_vs_obs_hypo"])),
                "ci95_lo": float(np.percentile(null_rs["pred_hypo_vs_obs_hypo"], 2.5)),
                "ci95_hi": float(np.percentile(null_rs["pred_hypo_vs_obs_hypo"], 97.5)),
            },
            "pred_hyper_vs_obs_hyper": {
                "mean_r": float(np.mean(null_rs["pred_hyper_vs_obs_hyper"])),
                "ci95_lo": float(np.percentile(null_rs["pred_hyper_vs_obs_hyper"], 2.5)),
                "ci95_hi": float(np.percentile(null_rs["pred_hyper_vs_obs_hyper"], 97.5)),
            },
            "n_trials": n_trials,
            "subtype_specific_hypo":  null_specific_hypo,
            "subtype_specific_hyper": null_specific_hyper,
        },
    }
    out_path = Path("outputs/logs/autism_subtypes_translation.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
