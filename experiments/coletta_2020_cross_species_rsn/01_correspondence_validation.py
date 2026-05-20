"""HOMER × Coletta 2020 cross-species RSN correspondence validation.

[Coletta et al. 2020, Sci Adv](https://www.science.org/doi/10.1126/sciadv.abb7187)
characterised mouse resting-state networks via group-ICA on mouse rsfMRI and
showed that a small set of mouse RSNs broadly correspond to canonical human
Yeo networks (Somatomotor, Visual, DMN, Salience, Limbic, etc.).

This is a stricter version of Pagani's Test 1 ("does HOMER's name-bridge hold")
in three ways:

  (A) **Labeled correspondence** — aggregate π over (HOMER mouse network ×
      Schaefer-Yeo7 human network), score diagonal-argmax, ratio over null.
      Re-runs the test with the canonical Yeo-7 partition (rather than
      Pagani's bespoke 8-net scheme) to check robustness of the bridge.

  (B) **Data-driven mouse RSNs via ICA** — Coletta's own methodology.
      Decompose the mouse FC matrix into independent components, route each
      through π, and ask which Yeo-7 network the predicted human spatial
      pattern best matches. Tests whether the cross-species correspondence
      survives when mouse networks are defined data-driven rather than from
      HOMER's anchor-derived PAIRID_TO_NETWORK.

  (C) **Network coherence (compactness)** — for each mouse network, how
      spatially compact is its predicted human-side image? A coherent mapping
      yields tight clusters; an incoherent one scatters mass across human
      space. Compared against permuted-π null.

Together these test whether HOMER's π preserves the cross-species network
structure under multiple operationalisations of what a "network" is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import FastICA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "autism_subtypes"))

from homer.data import load_cached
from homer.data.networks import PAIRID_TO_NETWORK, NETWORKS, assign_networks
from homer.data.anchors import get_anchor_index


# ============================================================================
# Sub-test A: labeled correspondence (HOMER mouse-nets × Schaefer-Yeo7 human-nets)
# ============================================================================


def labeled_correspondence(pi, mouse_net, mouse_names, human_net, human_names,
                            target_pairs):
    """Build N[i,j] = Σ_{m∈net_i, h∈net_j} π[m,h] then row-normalise.
    Score each target pair (mouse_name, human_name): is the diagonal the
    argmax of its row?
    """
    n_m_nets = len(mouse_names)
    n_h_nets = len(human_names)
    N = np.zeros((n_m_nets, n_h_nets))
    for mi in range(n_m_nets):
        m_mask = mouse_net == mi
        if not m_mask.any(): continue
        for hj in range(n_h_nets):
            h_mask = human_net == hj
            if not h_mask.any(): continue
            N[mi, hj] = pi[np.ix_(m_mask, h_mask)].sum()
    row_sum = N.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    N_norm = N / row_sum
    col_sum = N.sum(axis=0); total = N.sum()
    col_frac = col_sum / total if total > 0 else col_sum

    results = []
    for m_name, h_name in target_pairs:
        if m_name not in mouse_names or h_name not in human_names:
            results.append({"mouse_net": m_name, "human_net": h_name,
                             "status": "missing"})
            continue
        i = mouse_names.index(m_name); j = human_names.index(h_name)
        if N[i].sum() == 0:
            results.append({"mouse_net": m_name, "human_net": h_name,
                             "status": "empty_mouse_net"})
            continue
        order = (-N[i, :]).argsort()
        rank = int(np.where(order == j)[0][0]) + 1
        expected = col_frac[j] if total > 0 else 1.0 / n_h_nets
        results.append({
            "mouse_net": m_name, "human_net": h_name, "rank": rank,
            "row_norm_mass": float(N_norm[i, j]),
            "expected_mass_null": float(expected),
            "ratio_over_null": float(N_norm[i, j] / max(expected, 1e-12)),
            "is_argmax_diagonal": bool(order[0] == j),
            "argmax_human_net": human_names[int(order[0])],
        })
    return N, N_norm, results


# ============================================================================
# Sub-test B: data-driven mouse RSNs via ICA
# ============================================================================


def ica_mouse_rsns(fc_mouse, n_components=7, top_pct=10.0, seed=42):
    """Decompose mouse FC into ICA spatial components.

    Procedure:
      1. Fisher-z + diagonal-blank
      2. Row-wise threshold (keep top top_pct)
      3. FastICA on the (n_parcels, n_features=n_parcels) matrix
         — each row treated as a "subject" view of the connectivity
      4. Sign-flip each component so the most-positive parcel is positive
    Returns (n_components, n_parcels) spatial maps.
    """
    fc = np.clip(fc_mouse, -0.9999, 0.9999).astype(np.float64)
    fcz = np.arctanh(fc)
    np.fill_diagonal(fcz, 0.0)
    # Threshold
    thresh = np.percentile(fcz, 100.0 - top_pct, axis=1, keepdims=True)
    fcz_thr = np.where(fcz >= thresh, fcz, 0.0)
    # ICA — each parcel's connectivity row treated as an observation
    ica = FastICA(n_components=n_components, random_state=seed, max_iter=2000)
    # FastICA expects (n_samples, n_features); we want each parcel to be a sample
    components = ica.fit_transform(fcz_thr).T   # (n_components, n_parcels)
    # Sign-flip so peak weight is positive
    for k in range(n_components):
        if abs(components[k].min()) > abs(components[k].max()):
            components[k] = -components[k]
    return components


def label_mouse_ica_by_anatomy(mouse_components, mouse_paper_net, mouse_names):
    """Label each mouse ICA component by the anatomical network its peak
    weights cluster in. Returns list of (component_idx, network_name)."""
    labels = []
    for k, comp in enumerate(mouse_components):
        # Find parcels with top 5% of weights
        top_mask = comp >= np.percentile(comp, 95)
        if not top_mask.any():
            labels.append((k, None)); continue
        # Which mouse_paper_net dominates among the top parcels?
        top_nets = mouse_paper_net[top_mask]
        unique, counts = np.unique(top_nets[top_nets >= 0], return_counts=True)
        if len(unique) == 0:
            labels.append((k, None)); continue
        dominant = unique[counts.argmax()]
        labels.append((k, mouse_names[dominant]))
    return labels


# ============================================================================
# Sub-test C: network coherence
# ============================================================================


def network_coherence(pi, mouse_net, n_mouse_nets, H_var):
    """For each mouse network, measure how compact its predicted human-side
    image is. Returns per-network mean centroid spread (mm).
    """
    h_xyz = H_var[["x", "y", "z"]].to_numpy()
    argmax_h = pi.argmax(axis=1)
    out = {}
    for mi in range(n_mouse_nets):
        m_mask = mouse_net == mi
        if not m_mask.any(): continue
        h_pts = h_xyz[argmax_h[m_mask]]
        centroid = h_pts.mean(axis=0)
        spread = float(np.linalg.norm(h_pts - centroid, axis=1).mean())
        out[mi] = {"n_mouse": int(m_mask.sum()),
                    "centroid_spread_mm": spread}
    return out


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 80)
    print("HOMER × Coletta 2020 — cross-species RSN correspondence validation")
    print("=" * 80)

    # ---- Load HOMER + atlas labels ----
    pi = np.load(ROOT / "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")
    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    print(f"  π: {pi.shape}, total mass {pi.sum():.4f}")

    # Mouse network assignment via HOMER's PAIRID_TO_NETWORK (anchor-derived)
    idx_m = get_anchor_index(M.var)
    mouse_net_homer = assign_networks(M.var, idx_m)
    print(f"  Mouse PAIRID_TO_NETWORK assigns 1864 parcels to {len(NETWORKS)} networks")

    # Human network via Schaefer-Yeo7 (audit-corrected version)
    from importlib import import_module
    nc = import_module("01_network_crossvalidation")
    human_net, human_paper_names = nc.assign_human_paper_networks(H.var, separate_aud=True)
    # Merge Auditory into SomatoMotor for canonical Yeo-7
    aud_idx = human_paper_names.index("Auditory")
    som_idx = human_paper_names.index("SomatoMotor")
    human_net = human_net.copy()
    human_net[human_net == aud_idx] = som_idx
    print(f"  Human Schaefer-Yeo7+Subcortical: {len(set(human_net.tolist()))} unique networks")

    # ========================================================================
    # Sub-test A: Labeled correspondence
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUB-TEST A — Labeled correspondence (HOMER mouse-nets × Yeo-7 human-nets)")
    print("=" * 80)

    # HOMER → Yeo-7 canonical pairings (best mapping each direction can be matched)
    target_pairs = [
        ("sensorimotor", "SomatoMotor"),
        ("visual",       "Visual"),
        ("auditory",     "Auditory"),   # technically merged into SomMot; tested separately
        ("salience",     "Salience"),
        ("frontal_dmn",  "DMN"),
        ("temporal_dmn", "DMN"),
        ("limbic",       "Limbic"),
        ("frontoparietal", "DorsAtten"),
        ("subcortical",  "Subcortical"),
        ("olfactory",    "Limbic"),
    ]
    N, N_norm, score_a = labeled_correspondence(
        pi, mouse_net_homer, NETWORKS, human_net, human_paper_names,
        target_pairs=target_pairs,
    )
    print(f"\n{'Pair':<35s} | {'mass on target':>16s} | {'null':>7s} | {'ratio':>7s} | argmax?")
    print("-" * 90)
    for r in score_a:
        if r.get("status"):
            print(f"  {r['mouse_net']} → {r['human_net']}: {r['status']}")
        else:
            star = "★" if r["is_argmax_diagonal"] else " "
            pair = f"{r['mouse_net']} → {r['human_net']}"
            print(f"  {pair:<33s} | {r['row_norm_mass']*100:>14.1f}% | "
                  f"{r['expected_mass_null']*100:>6.1f}% | "
                  f"{r['ratio_over_null']:>6.2f}× | {star}")
    n_diag = sum(r.get("is_argmax_diagonal", False) for r in score_a)
    n_pairs = sum("ratio_over_null" in r for r in score_a)
    print(f"\n  → {n_diag}/{n_pairs} canonical pairs diagonal-argmax")

    # ========================================================================
    # Sub-test B: Data-driven mouse RSNs via ICA
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUB-TEST B — Data-driven mouse RSNs via ICA on mouse FC")
    print("=" * 80)

    fc_mouse = M.uns["fc_mean"]
    n_ica = 7  # Coletta's order of magnitude
    print(f"\nDecomposing mouse FC into {n_ica} ICA components...")
    mouse_components = ica_mouse_rsns(fc_mouse, n_components=n_ica, top_pct=10.0)
    print(f"  components shape: {mouse_components.shape}")

    ica_labels = label_mouse_ica_by_anatomy(
        mouse_components, mouse_net_homer, NETWORKS)
    print(f"\nLabeling each ICA component by dominant anatomical network:")
    for k, lbl in ica_labels:
        peak_n = np.sum(mouse_components[k] >= np.percentile(mouse_components[k], 95))
        print(f"  IC{k}: dominant network = {lbl}  (peak parcels: {peak_n})")

    # For each ICA component, route through π → predicted human spatial map
    # Then aggregate to Yeo-7 networks and find which one is the argmax
    print(f"\nFor each mouse IC, route through π and find best-match Yeo-7 network:")
    ica_correspondence = []
    for k in range(n_ica):
        pred_h = mouse_components[k] @ pi   # (2094,)
        # Mean per Yeo-7 network
        per_net = []
        net_names_seen = []
        for hj in range(len(human_paper_names)):
            mask = human_net == hj
            if mask.any():
                per_net.append(pred_h[mask].mean())
                net_names_seen.append(human_paper_names[hj])
        per_net = np.array(per_net)
        argmax_net = net_names_seen[int(np.argmax(per_net))]
        ica_correspondence.append({
            "ic": k,
            "mouse_label": ica_labels[k][1],
            "best_match_yeo7": argmax_net,
            "second_match_yeo7": net_names_seen[int(np.argsort(-per_net)[1])],
            "per_net_means": dict(zip(net_names_seen, per_net.tolist())),
        })
        print(f"  IC{k} ({ica_labels[k][1]:<14s}) → predicted human argmax: {argmax_net}")

    # ========================================================================
    # Sub-test C: Network coherence (compactness in human space)
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUB-TEST C — Network coherence (spatial compactness in human space)")
    print("=" * 80)

    real_coh = network_coherence(pi, mouse_net_homer, len(NETWORKS), H.var)
    # Permuted-π null
    rng = np.random.default_rng(seed=42)
    n_trials = 100
    null_spread = {mi: [] for mi in real_coh}
    for _ in range(n_trials):
        perm = rng.permutation(pi.shape[0])
        pi_n = pi[perm]
        coh_n = network_coherence(pi_n, mouse_net_homer, len(NETWORKS), H.var)
        for mi in real_coh:
            if mi in coh_n:
                null_spread[mi].append(coh_n[mi]["centroid_spread_mm"])
    print(f"\n{'Network':<18s} | {'n_mouse':>8s} | {'centroid spread (mm)':>22s} | "
          f"{'null spread (95% CI)':>26s} | {'ratio':>6s}")
    print("-" * 100)
    coh_results = []
    for mi in sorted(real_coh):
        net_name = NETWORKS[mi]
        real = real_coh[mi]["centroid_spread_mm"]
        n_m = real_coh[mi]["n_mouse"]
        nulls = np.array(null_spread[mi])
        null_mean = nulls.mean()
        ci = (np.percentile(nulls, 2.5), np.percentile(nulls, 97.5))
        ratio = real / null_mean if null_mean > 0 else float("nan")
        # ratio < 1 means HOMER more compact than null (good)
        print(f"  {net_name:<16s} | {n_m:>8d} | {real:>20.1f} | "
              f"{null_mean:>10.1f} ({ci[0]:.0f}, {ci[1]:.0f})       | {ratio:>5.2f}")
        coh_results.append({
            "network": net_name, "n_mouse": n_m,
            "real_spread_mm": real, "null_mean_mm": null_mean,
            "null_ci95_mm": list(ci),
            "ratio_real_over_null": ratio,
            "more_compact_than_null": bool(real < null_mean),
        })

    n_more_compact = sum(1 for r in coh_results if r["more_compact_than_null"])
    print(f"\n  → HOMER's mouse-network images are MORE compact than null in "
          f"{n_more_compact}/{len(coh_results)} networks")

    # Save
    out = {
        "pi_file": "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy",
        "sub_test_A_labeled_correspondence": {
            "mouse_networks": list(NETWORKS),
            "human_networks": list(human_paper_names),
            "target_pairs":   [list(t) for t in target_pairs],
            "per_pair_scores": score_a,
            "n_diagonal_argmax": int(n_diag),
            "n_pairs_scored":   int(n_pairs),
        },
        "sub_test_B_ica_data_driven": {
            "n_components": n_ica,
            "labels":       [{"ic": k, "mouse_label": l} for k, l in ica_labels],
            "correspondence": ica_correspondence,
        },
        "sub_test_C_network_coherence": {
            "n_null_trials": n_trials,
            "per_network": coh_results,
            "n_networks_more_compact_than_null": int(n_more_compact),
        },
    }
    out_path = ROOT / "outputs" / "logs" / "coletta_2020_cross_species_rsn.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
