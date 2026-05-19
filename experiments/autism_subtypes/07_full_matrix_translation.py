"""Test 2c — Full per-network-pair Δ-matrix translation through π.

Sharper version of Test 2b: instead of collapsing each subtype matrix to per-network
intensity (a length-9 / length-8 vector), use the full per-network-PAIR perturbation
matrix. This gives ~36 paired upper-triangle entries (8×8 → 36 unique under symmetry)
instead of 8 row-sums.

Procedure:
  1. Read mouse 9×9 hypo + hyper matrices; symmetrize; compute Δ_mouse = hyper − hypo.
  2. Read human 8×8 hypo + hyper matrices; symmetrize; compute Δ_human_obs = hyper − hypo.
  3. Build a translation operator T (n_mouse_net × n_human_net) from π, where
     T[mi, hi] = P(human net hi | mouse net mi) — row-normalised aggregated π.
  4. Predicted Δ_human[hi, hj] = Σ_mi Σ_mj T[mi, hi] · T[mj, hj] · Δ_mouse[mi, mj]
     (T · Δ_mouse · T.T using matrix form).
  5. Correlate predicted vs observed over the 36 upper-triangle entries.
  6. Permuted-π null: shuffle π rows, repeat 200 trials.

Critically, we map mouse parcels to Pagani's *9* mouse networks (separating
Caudate Putamen from Thalamus, which HOMER's PAIRID_TO_NETWORK lumps as
"subcortical"). We do this via per-parcel nearest-Garin-anchor assignment using
the same xyz logic in homer.data.networks.assign_networks, but exposing the
*anchor pair_id* rather than the network name.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
nc = import_module("01_network_crossvalidation")
st = import_module("04_subtype_translation")
assign_human_paper_networks = nc.assign_human_paper_networks
load_pagani_subtype_matrices = st.load_pagani_subtype_matrices

from homer.data import load_cached
from homer.data.anchors import get_anchor_index


# Map Garin pair_id → Pagani 9-net name
# (Pagani's 9-net mouse atlas + how each Garin anchor pid maps in)
GARIN_PID_TO_PAGANI_MOUSE = {
    # Pagani: Auditory, BF, Caudate Putamen, DMN, HC, Salience, Somatomotor, Thalamus, Visual
    1:  "DMN",             # mPFC
    2:  "Somatomotor",     # Motor / premotor
    3:  "Somatomotor",     # Somatosensory (S1) — collapse to Somatomotor like the paper does
    4:  "DMN",             # Posterior parietal (parietal cortex; in mouse FC, often DMN/PCC-aligned)
    5:  "Visual",          # V1
    6:  "Visual",          # V2 / extrastriate
    7:  "Auditory",        # Auditory cortex
    8:  "DMN",             # MIPT (medial temporal, DMN-aligned)
    9:  "Salience",        # Insula
    10: "BF",              # Septum
    11: "BF",              # Olfactory cortex (piriform/AON) — Pagani's BF is broader basal-forebrain
    12: "HC",              # Periarchicortex (hippocampal allocortex)
    13: "Caudate Putamen", # Striatum (CPu) ← KEY for splitting subcortical
    14: "BF",              # Basal forebrain (NBM)
    15: "Caudate Putamen", # Pallidum (sits at ventral striatum / GP); Pagani lumps into "CaudatePutamen"
    16: "Salience",        # Claustrum
    17: "HC",              # Amygdala (limbic; Pagani lumps into HC)
    18: "Thalamus",        # Hypothalamus → Thalamus (Pagani has no hypothal; nearest is Thalamus)
    19: "Thalamus",        # Thalamus ← KEY for splitting subcortical
    20: None,              # Pons — Pagani has no brainstem
    21: None,              # Tectum — Pagani has no brainstem
}


def assign_mouse_pagani_networks(M_var: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Assign each of 1864 mouse parcels to one of Pagani's 9 mouse networks
    (plus a -1 sentinel for parcels that don't map cleanly — Pons/Tectum).

    Uses the same nearest-anchor-by-normalised-xyz logic as
    homer.data.networks.assign_networks, but maps to Pagani names via
    GARIN_PID_TO_PAGANI_MOUSE so we can split Caudate Putamen vs Thalamus.
    """
    idx_m = get_anchor_index(M_var)
    coords = M_var[["x", "y", "z"]].values.astype(np.float64)
    lo = coords.min(0, keepdims=True); hi = coords.max(0, keepdims=True)
    cn = (coords - lo) / np.maximum(hi - lo, 1e-9)
    anchor_pos = idx_m.pos
    anchor_xyz = cn[anchor_pos]
    # For each parcel, nearest anchor
    sq_a = (anchor_xyz**2).sum(1, keepdims=True)
    sq_b = (cn**2).sum(1, keepdims=True)
    d2 = sq_b + sq_a.T - 2.0 * cn @ anchor_xyz.T
    nearest_k = d2.argmin(axis=1)
    nearest_pid = np.array([int(idx_m.pair_ids[k]) for k in nearest_k])
    # Override: if a parcel IS an anchor, use its own pid
    is_anchor = np.zeros(len(M_var), dtype=bool)
    is_anchor[anchor_pos] = True
    for k, pos in enumerate(anchor_pos):
        nearest_pid[pos] = int(idx_m.pair_ids[k])

    pagani_names = ["Auditory", "BF", "Caudate Putamen", "DMN", "HC",
                    "Salience", "Somatomotor", "Thalamus", "Visual"]
    name_to_idx = {n: i for i, n in enumerate(pagani_names)}
    out = np.full(len(M_var), -1, dtype=np.int32)
    for p in range(len(M_var)):
        pname = GARIN_PID_TO_PAGANI_MOUSE.get(int(nearest_pid[p]))
        if pname is not None:
            out[p] = name_to_idx[pname]
    return out, pagani_names


def symmetrise(M: np.ndarray) -> np.ndarray:
    return (M + M.T) / 2.0


def build_translation_operator(pi: np.ndarray,
                                 mouse_net: np.ndarray, n_mouse_nets: int,
                                 human_net: np.ndarray, n_human_nets: int,
                                 ) -> np.ndarray:
    """T[mi, hj] = P(human net hj | mouse net mi) by aggregating π."""
    N = np.zeros((n_mouse_nets, n_human_nets))
    for mi in range(n_mouse_nets):
        mm = mouse_net == mi
        if not mm.any():
            continue
        for hj in range(n_human_nets):
            hm = human_net == hj
            if not hm.any():
                continue
            N[mi, hj] = pi[np.ix_(mm, hm)].sum()
    # Row-normalize → conditional distribution
    row_sum = N.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return N / row_sum


def upper_triangle_flat(M: np.ndarray) -> np.ndarray:
    """Return upper triangle + diagonal as a 1D vector (n × (n+1) / 2)."""
    iu = np.triu_indices(M.shape[0])
    return M[iu]


def main():
    print("=" * 80)
    print("Pagani 2026 Test 2c — full 64-element matrix translation through π")
    print("=" * 80)

    M, _ = load_cached("mouse", cache_dir="outputs/anndata")
    H, _ = load_cached("human", cache_dir="outputs/anndata")
    pi = np.load("outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")
    print(f"π: {pi.shape}, total mass: {pi.sum():.4f}")

    mouse_net, mouse_pagani_names = assign_mouse_pagani_networks(M.var)
    n_unassigned = int((mouse_net < 0).sum())
    print(f"\nMouse parcels per Pagani net (drop {n_unassigned} brainstem/tectum):")
    for i, n in enumerate(mouse_pagani_names):
        print(f"  {n:18s}: {(mouse_net == i).sum()}")

    human_net, human_paper_names = assign_human_paper_networks(H.var, separate_aud=True)
    aud_idx = human_paper_names.index("Auditory")
    som_idx = human_paper_names.index("SomatoMotor")
    human_net = human_net.copy()
    human_net[human_net == aud_idx] = som_idx
    # Build a contiguous 8-net partition matching Pagani's order:
    pagani_human_names = ["Control", "DMN", "DorsAtten", "Limbic",
                          "Salience", "SomatoMotor", "Visual", "Subcortical"]
    # Map our internal-name idx → pagani idx
    h_name_to_idx = {n: human_paper_names.index(n) for n in pagani_human_names}
    new_human_net = np.full_like(human_net, -1)
    for new_i, n in enumerate(pagani_human_names):
        new_human_net[human_net == h_name_to_idx[n]] = new_i
    print(f"\nHuman parcels per Pagani net:")
    for i, n in enumerate(pagani_human_names):
        print(f"  {n:18s}: {(new_human_net == i).sum()}")

    # Drop unassigned mouse parcels (and corresponding π rows) for the test
    keep_m = mouse_net >= 0
    pi_kept = pi[keep_m]
    # Re-normalize column-marginal of kept rows so the operator stays a proper distribution
    print(f"\nUsing {keep_m.sum()}/{len(keep_m)} mouse parcels; total π mass after drop: {pi_kept.sum():.3f}")

    # Build translation operator T
    T = build_translation_operator(
        pi_kept, mouse_net[keep_m], len(mouse_pagani_names),
        new_human_net, len(pagani_human_names),
    )
    print(f"\nTranslation operator T ({len(mouse_pagani_names)} x {len(pagani_human_names)}) — "
          f"each row sums to ~1:")
    print(f"  {' '*18}" + " ".join(f"{n[:6]:>6s}" for n in pagani_human_names))
    for i, mn in enumerate(mouse_pagani_names):
        row = " ".join(f"{T[i,j]:6.3f}" for j in range(len(pagani_human_names)))
        print(f"  {mn:<18s}{row}")

    # Load Pagani matrices
    data = load_pagani_subtype_matrices()
    M_hypo  = symmetrise(data["mouse_hypo"])
    M_hyper = symmetrise(data["mouse_hyper"])
    H_hypo_obs  = symmetrise(data["human_hypo"])
    H_hyper_obs = symmetrise(data["human_hyper"])

    # Order rows/cols of Pagani matrices to match our mouse_pagani_names + pagani_human_names
    # — Pagani's order matches ours by construction.
    Δ_mouse = M_hyper - M_hypo                  # 9×9 signed
    Δ_human_obs = H_hyper_obs - H_hypo_obs      # 8×8 signed

    # Predict: T^T · Δ_mouse · T
    Δ_human_pred = T.T @ Δ_mouse @ T

    print(f"\nMouse Δ (hyper − hypo) 9×9 first 3 rows:")
    for r in range(3):
        print(f"  {mouse_pagani_names[r]:18s}: {Δ_mouse[r]}")
    print(f"\nObserved human Δ 8×8 first 3 rows:")
    for r in range(3):
        print(f"  {pagani_human_names[r]:18s}: {[f'{v:+.2f}' for v in Δ_human_obs[r]]}")
    print(f"\nPredicted human Δ 8×8 first 3 rows:")
    for r in range(3):
        print(f"  {pagani_human_names[r]:18s}: {[f'{v:+.4f}' for v in Δ_human_pred[r]]}")

    # Upper-triangle flatten and correlate
    pred_flat = upper_triangle_flat(Δ_human_pred)
    obs_flat = upper_triangle_flat(Δ_human_obs)
    print(f"\nUsing {len(pred_flat)} upper-triangle (incl. diagonal) elements of 8×8 matrices.")

    r_p, p_p = pearsonr(pred_flat, obs_flat)
    r_s, p_s = spearmanr(pred_flat, obs_flat)
    print(f"  Pearson  r = {r_p:+.3f} (analytical p = {p_p:.4f})")
    print(f"  Spearman ρ = {r_s:+.3f} (analytical p = {p_s:.4f})")

    # Permuted-π null
    print(f"\nPermuted-π null (200 trials):")
    rng = np.random.default_rng(seed=42)
    n_trials = 200
    null_p = []
    null_s = []
    for _ in range(n_trials):
        perm = rng.permutation(pi.shape[0])
        pi_n = pi[perm]
        pi_n_kept = pi_n[keep_m]
        T_n = build_translation_operator(pi_n_kept, mouse_net[keep_m],
                                          len(mouse_pagani_names),
                                          new_human_net, len(pagani_human_names))
        Δ_pred_n = T_n.T @ Δ_mouse @ T_n
        pred_n = upper_triangle_flat(Δ_pred_n)
        rpn, _ = pearsonr(pred_n, obs_flat)
        rsn, _ = spearmanr(pred_n, obs_flat)
        null_p.append(rpn); null_s.append(rsn)
    null_p = np.array(null_p); null_s = np.array(null_s)
    p_one_sided_p = float(np.mean(null_p >= r_p))
    p_one_sided_s = float(np.mean(null_s >= r_s))
    print(f"  Pearson  null mean={null_p.mean():+.3f}, "
          f"95% CI ({np.percentile(null_p, 2.5):+.3f}, {np.percentile(null_p, 97.5):+.3f}), "
          f"  one-sided p = {p_one_sided_p:.3f}")
    print(f"  Spearman null mean={null_s.mean():+.3f}, "
          f"95% CI ({np.percentile(null_s, 2.5):+.3f}, {np.percentile(null_s, 97.5):+.3f}), "
          f"  one-sided p = {p_one_sided_s:.3f}")

    # Cell-by-cell agreement
    same_sign = (np.sign(pred_flat) == np.sign(obs_flat)).sum()
    print(f"\nCell-by-cell: {same_sign}/{len(pred_flat)} matrix entries have the same sign.")

    out = {
        "pi_file": "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy",
        "n_elements": int(len(pred_flat)),
        "pearson_r": float(r_p), "pearson_p_analytical": float(p_p),
        "spearman_r": float(r_s), "spearman_p_analytical": float(p_s),
        "null": {
            "n_trials": n_trials,
            "pearson_mean": float(null_p.mean()),
            "pearson_ci95": [float(np.percentile(null_p, 2.5)), float(np.percentile(null_p, 97.5))],
            "spearman_mean": float(null_s.mean()),
            "spearman_ci95": [float(np.percentile(null_s, 2.5)), float(np.percentile(null_s, 97.5))],
            "pearson_empirical_p": p_one_sided_p,
            "spearman_empirical_p": p_one_sided_s,
        },
        "same_sign_fraction": float(same_sign / len(pred_flat)),
        "delta_human_observed_flat":  obs_flat.tolist(),
        "delta_human_predicted_flat": pred_flat.tolist(),
        "mouse_pagani_names": mouse_pagani_names,
        "human_pagani_names": pagani_human_names,
        "translation_operator": T.tolist(),
    }
    out_path = Path("outputs/logs/autism_subtypes_full_matrix.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
