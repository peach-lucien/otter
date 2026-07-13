"""Discrete reframes of the Margulies/Huntenburg principal-gradient test.

The continuous correlation (routed mouse gradient vs human gradient) is |r|=0.40
but does NOT survive a spatial spin null (p=0.16): two smooth monotone maps
correlate by spatial autocorrelation alone (see 01_gradient_validation.py and
experiments/spatial_null_check/). HOMER's *strong* mode is discrete correspondence
(the network bridge and Coletta RSNs survive spin), so here we ask the gradient
question categorically, does π preserve the discrete content of the gradient?
and test it against the fair spin null.

Two tests:
  A. Gradient-TIER classification. Bin the human gradient into 3 tiers
     (unimodal → transmodal). Route the mouse gradient through π, bin the
     prediction by the human tier edges, and measure exact + adjacent tier
     accuracy. Null = spin the mouse gradient on the mouse sphere, route through
     the REAL π, re-bin, recompute accuracy (the fair translation-spin null).
  B. Network RANK-ORDER. Order the mouse networks by their mean mouse-gradient
     and the human networks by their mean human-gradient; route the per-network
     mouse gradient through the network-aggregated π and ask whether the
     predicted human-network ordering matches the observed one (Spearman over
     the networks). This ties the gradient to the network bridge that already
     survives spin. Null = spin mouse parcels, recompute, re-rank.

Usage:
    PYTHONPATH=src python experiments/margulies_2016_principal_gradient/03_discrete_reframe.py
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "autism_subtypes"))

from homer.data import load_cached                       # noqa: E402
from homer.eval.nulls import _haar_rotation, _route_normalized  # noqa: E402

grad_mod = import_module("01_gradient_validation")
nc = import_module("01_network_crossvalidation")


def tiers(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Tier index 0..len(edges) for finite vals; -1 for NaN."""
    out = np.full(vals.shape, -1, dtype=int)
    fin = np.isfinite(vals)
    out[fin] = np.digitize(vals[fin], edges)
    return out


def tier_accuracy(pred_vals, human_tier, edges):
    """Exact and adjacent (off-by-one) tier accuracy of pred vs human tiers."""
    pred_tier = tiers(pred_vals, edges)
    m = (pred_tier >= 0) & (human_tier >= 0)
    exact = float((pred_tier[m] == human_tier[m]).mean())
    adj = float((np.abs(pred_tier[m] - human_tier[m]) <= 1).mean())
    return exact, adj, int(m.sum())


def main():
    print("=" * 78)
    print("Margulies gradient. DISCRETE reframes vs a fair spin null")
    print("=" * 78)

    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = np.load(ROOT / "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")
    mouse_coords = M.var[["x", "y", "z"]].to_numpy(float)

    # Read the gradients from 01_gradient_validation.py's output rather than recomputing.
    # principal_gradient() now requires an external hierarchy reference to select the
    # correct diffusion component (it used to take the wrong one; see its docstring),
    # so recomputing here would duplicate that logic and risk it drifting out of sync.
    _g = json.loads((ROOT / "outputs/logs/margulies_2016_gradient.json").read_text())
    mouse_grad = np.asarray(_g["mouse_gradient"], float)
    human_grad = np.asarray(_g["human_gradient"], float)
    print(f"  using components mouse={_g['component_selection']['mouse']['selected_component']}, "
          f"human={_g['component_selection']['human']['selected_component']} "
          f"(selected against each species' T1w/T2w map)")
    pred = _route_normalized(mouse_grad, pi)
    # resolve eigenvector sign against the observed human gradient
    mfin = np.isfinite(pred) & np.isfinite(human_grad)
    if np.corrcoef(pred[mfin], human_grad[mfin])[0, 1] < 0:
        pred = -pred
        mouse_grad = -mouse_grad

    # ---------- Test A: gradient-tier classification ----------
    NT = 3
    edges = np.nanpercentile(human_grad, [100 / NT * k for k in range(1, NT)])
    human_tier = tiers(human_grad, edges)
    exact, adj, n = tier_accuracy(pred, human_tier, edges)
    chance = 1.0 / NT

    # spin null: rotate mouse gradient on the mouse sphere, route through real π
    c = mouse_coords - np.nanmean(mouse_coords, 0)
    sph = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)
    rng = np.random.default_rng(0)
    NSPIN = 1000
    null_exact = np.empty(NSPIN)
    null_adj = np.empty(NSPIN)
    for t in range(NSPIN):
        rot = sph @ _haar_rotation(rng).T
        _, perm = cKDTree(rot).query(sph)
        p_spun = _route_normalized(mouse_grad[perm], pi)
        e, a, _ = tier_accuracy(p_spun, human_tier, edges)
        null_exact[t] = e
        null_adj[t] = a
    p_exact = (np.sum(null_exact >= exact) + 1) / (NSPIN + 1)
    p_adj = (np.sum(null_adj >= adj) + 1) / (NSPIN + 1)

    print(f"\n[A] Gradient-tier classification ({NT} tiers, n={n} human parcels):")
    print(f"    exact accuracy = {exact:.3f}  (chance {chance:.3f}; spin null mean "
          f"{null_exact.mean():.3f}) → spin p = {p_exact:.3f}")
    print(f"    adjacent (±1)  = {adj:.3f}  (spin null mean {null_adj.mean():.3f}) "
          f"→ spin p = {p_adj:.3f}")

    # ---------- Test B: network rank-order ----------
    mnet, mnames = nc.assign_mouse_paper_networks(M.var)
    hnet, hnames = nc.assign_human_paper_networks(H.var)

    def net_means(vals, net, n_names):
        out = np.full(n_names, np.nan)
        for k in range(n_names):
            sel = (net == k) & np.isfinite(vals)
            if sel.any():
                out[k] = vals[sel].mean()
        return out

    obs_h_net = net_means(human_grad, hnet, len(hnames))
    # route per-network mouse gradient through π aggregated to networks
    mouse_net_grad_parcel = np.full(len(mouse_grad), np.nan)
    mm = net_means(mouse_grad, mnet, len(mnames))
    for k in range(len(mnames)):
        mouse_net_grad_parcel[mnet == k] = mm[k]
    pred_h_parcel = _route_normalized(mouse_net_grad_parcel, pi)
    pred_h_net = net_means(pred_h_parcel, hnet, len(hnames))

    valid = np.isfinite(obs_h_net) & np.isfinite(pred_h_net)
    rho_net = float(spearmanr(obs_h_net[valid], pred_h_net[valid])[0])

    # spin null for the network rank-order
    null_rho = np.empty(NSPIN)
    rng2 = np.random.default_rng(1)
    for t in range(NSPIN):
        rot = sph @ _haar_rotation(rng2).T
        _, perm = cKDTree(rot).query(sph)
        spun_parcel = mouse_net_grad_parcel[perm]
        ph = net_means(_route_normalized(spun_parcel, pi), hnet, len(hnames))
        v = np.isfinite(obs_h_net) & np.isfinite(ph)
        null_rho[t] = spearmanr(obs_h_net[v], ph[v])[0] if v.sum() >= 3 else np.nan
    p_rho = (np.sum(np.abs(null_rho) >= abs(rho_net)) + 1) / (NSPIN + 1)

    print(f"\n[B] Network rank-order ({int(valid.sum())} human networks):")
    print(f"    Spearman ρ(predicted, observed network gradient) = {rho_net:+.3f}")
    print(f"    spin null |ρ| mean {np.nanmean(np.abs(null_rho)):.3f} → spin p = {p_rho:.3f}")
    order_obs = [hnames[i] for i in np.argsort(np.where(valid, obs_h_net, np.inf)) if valid[i]]
    order_pred = [hnames[i] for i in np.argsort(np.where(valid, pred_h_net, np.inf)) if valid[i]]
    print(f"    observed  unimodal→transmodal: {order_obs}")
    print(f"    predicted unimodal→transmodal: {order_pred}")

    out = {
        "pi_file": "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy",
        "continuous_reference": {"abs_pearson_r": 0.402, "spin_p": 0.16},
        "tier_classification": {
            "n_tiers": NT, "n_parcels": n, "exact_accuracy": exact,
            "adjacent_accuracy": adj, "chance": chance,
            "spin_null_exact_mean": float(null_exact.mean()),
            "spin_null_adjacent_mean": float(null_adj.mean()),
            "spin_p_exact": float(p_exact), "spin_p_adjacent": float(p_adj),
            "n_spin": NSPIN,
        },
        "network_rank_order": {
            "n_networks": int(valid.sum()),
            "spearman_rho": rho_net,
            "spin_null_abs_rho_mean": float(np.nanmean(np.abs(null_rho))),
            "spin_p": float(p_rho),
            "observed_order": order_obs, "predicted_order": order_pred,
        },
    }
    out_path = ROOT / "outputs" / "logs" / "margulies_discrete_reframe.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
