"""Network-level cross-species validation against Pagani 2026.

Pagani et al. (2026), *Nat Neurosci*, "Autism subtypes identified using cross-species
functional connectivity analyses", establish a mouse↔human translation that operates at
**network level**. They define 9 mouse networks (Auditory, BF, Caudate Putamen, DMN,
HC, Salience, Somatomotor, Thalamus, Visual; ED Fig 1) and 8 human networks (Control,
DMN, DorsAtten, Limbic, Salience, SomatoMotor, Visual, Subcortical; Fig 4e). Their
cross-species correspondence is **by name**. Somatomotor in mouse ↔ Somatomotor in
human, etc.

HOMER provides a quantitative π (1864 mouse × 2094 human) that can be aggregated to a
network-network mapping matrix. This script asks: **does π preferentially route mass
between like-named networks?**

If yes, HOMER's structural+anchor evidence corroborates the name-based correspondence
the paper relies on, providing a quantitative bridge for the paper's workflow.
If no, the name-based shortcut is over-confident in places HOMER thinks the structural
evidence disagrees.

Method:
  1. Assign each mouse parcel to a network using PAIRID_TO_NETWORK (nearest-anchor
     inheritance over Garin 21).
  2. Assign each human parcel to a Yeo-7 network using the Schaefer-400 17-network
     atlas, collapsed 17→7 with "Subcortical" for un-assigned parcels.
  3. Build harmonized 7-network labels common to both species:
       DMN, SomatoMotor, Visual, Auditory, Salience, Limbic, Subcortical
     (+ unmatched mouse-only networks: Brainstem, Frontoparietal; held out)
  4. Compute N[i, j] = Σ π[m, h] for m in mouse-net i, h in human-net j.
  5. Score diagonal-dominance: trace(N̂) where N̂ is row-normalised, and per-network
     "is the diagonal the argmax?" check.
  6. Compare against expected-by-area baseline (Σ π[m,*] is uniform 1/1864 in our
     setup, so column-sum is just human-net size, the question is whether mouse-net
     i preferentially loads on human-net j=i vs other human-nets weighted by their
     sizes).

Usage:
    PYTHONPATH=src python experiments/autism_subtypes/01_network_crossvalidation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from homer.data import load_cached, load_pi, pi_provenance
from homer.data.anchors import get_anchor_index
from homer.data.atlas_regions import (
    ATLAS_PATHS,
    assign_atlas_labels,
    assign_atlas_labels_with_hemisphere,
)
from homer.data.networks import PAIRID_TO_NETWORK, NETWORKS, assign_networks


# Mapping: HOMER's 11-network mouse scheme → Pagani paper's network names
# (matches paper's ED Fig 1 nine-network scheme + collapses some mouse-specific
# distinctions that don't have a clean human counterpart).
HOMER_NET_TO_PAPER_MOUSE: dict[str, str] = {
    "auditory":      "Auditory",
    "sensorimotor":  "SomatoMotor",
    "visual":        "Visual",
    "frontal_dmn":   "DMN",
    "temporal_dmn":  "DMN",
    "salience":      "Salience",
    "limbic":        "HC_Limbic",      # mouse limbic ≈ paper HC (hippocampal/limbic)
    "olfactory":     "BF_Olfactory",   # mouse olfactory ≈ paper BF (basal forebrain)
    "subcortical":   "Subcortical",    # paper splits to CaudPut/Thalamus/BF
    "frontoparietal": "Frontoparietal",  # mouse-specific; no clean human counterpart
    "brainstem":     "Brainstem",     # excluded from paper's networks
}

# Mapping: Schaefer-17 prefix → Yeo-7 network name (paper's human side)
SCHAEFER17_TO_YEO7: dict[str, str] = {
    "VisCent":     "Visual",
    "VisPeri":     "Visual",
    "SomMotA":     "SomatoMotor",
    "SomMotB":     "SomatoMotor",   # SomMotB_Aud + SomMotB_S2. Yeo-7 collapses auditory
    "DorsAttnA":   "DorsAtten",
    "DorsAttnB":   "DorsAtten",
    "SalVentAttnA": "Salience",
    "SalVentAttnB": "Salience",
    "Limbic":      "Limbic",
    "ContA":       "Control",
    "ContB":       "Control",
    "ContC":       "Control",
    "DefaultA":    "DMN",
    "DefaultB":    "DMN",
    "DefaultC":    "DMN",
    "TempPar":     "DMN",           # 17-net specific; Yeo-7 places it under Default
}

# Optional fine-grained version: Schaefer SomMotB_Aud → Auditory
# (lets us distinguish auditory cortex from primary somatomotor on the human side
# to match the mouse-side Auditory vs SomatoMotor distinction)
SCHAEFER_AUD_PARCELS = "SomMotB_Aud"  # substring that identifies auditory parcels


def _load_schaefer17_labels() -> dict[int, str]:
    """Return dict mapping Schaefer-400 ID → 17-network name prefix.
    For LH IDs 1-200 and RH IDs 201-400 (the official ordering).
    """
    order_path = Path("outputs/anndata/_schaefer_order.txt")
    if not order_path.exists():
        raise FileNotFoundError(
            f"{order_path} missing, extract from "
            f"data_external/p6ebec-hbp-d000038_SC-FC_HCP_eNKI_pub/"
            f"Schaefer2018_400Parcels_17Networks.zip first."
        )
    out: dict[int, str] = {}
    with order_path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            sid = int(parts[0])
            name = parts[1]            # e.g. "17Networks_LH_VisCent_ExStr_1"
            # Strip "17Networks_LH_" or "17Networks_RH_" prefix
            net = name.split("_", 2)[2]  # e.g. "VisCent_ExStr_1"
            out[sid] = net
    return out


def _yeo7_name_for_schaefer(schaefer_label: str, *, separate_aud: bool) -> str:
    """Look up Yeo-7 network for a Schaefer-17 label, optionally splitting auditory."""
    if separate_aud and SCHAEFER_AUD_PARCELS in schaefer_label:
        return "Auditory"
    for prefix, yeo7 in SCHAEFER17_TO_YEO7.items():
        if schaefer_label.startswith(prefix):
            return yeo7
    raise ValueError(f"Unmapped Schaefer label: {schaefer_label}")


def assign_human_paper_networks(H_var: pd.DataFrame, *, separate_aud: bool = True
                                 ) -> tuple[np.ndarray, list[str]]:
    """Assign each of 2094 human parcels to one of:
       Visual, SomatoMotor, DorsAtten, Salience, Limbic, Control, DMN, Auditory, Subcortical

    Strategy: use Schaefer-400 atlas labels (1-400). For Schaefer-assigned parcels,
    map 17-network ID → Yeo-7 name (and optionally split SomMotB_Aud as "Auditory").
    Parcels with no Schaefer ID (0) are assigned "Subcortical".
    """
    schaefer_labels = _load_schaefer17_labels()
    schaefer_ids = assign_atlas_labels(H_var, "schaefer_400", ATLAS_PATHS["schaefer_400"])
    schaefer_ids = assign_atlas_labels_with_hemisphere(H_var, schaefer_ids)

    if separate_aud:
        net_names = ["Visual", "Auditory", "SomatoMotor", "DorsAtten", "Salience",
                     "Limbic", "Control", "DMN", "Subcortical"]
    else:
        net_names = ["Visual", "SomatoMotor", "DorsAtten", "Salience",
                     "Limbic", "Control", "DMN", "Subcortical"]
    net_to_idx = {n: i for i, n in enumerate(net_names)}

    out = np.full(len(H_var), -1, dtype=np.int32)
    for p, sid in enumerate(schaefer_ids):
        if sid == 0:
            out[p] = net_to_idx["Subcortical"]
        else:
            yeo7 = _yeo7_name_for_schaefer(schaefer_labels[int(sid)], separate_aud=separate_aud)
            out[p] = net_to_idx[yeo7]
    assert (out >= 0).all()
    return out, net_names


def assign_mouse_paper_networks(M_var: pd.DataFrame, *, separate_aud: bool = True
                                 ) -> tuple[np.ndarray, list[str]]:
    """Assign each of 1864 mouse parcels to a paper-aligned network name.

    Uses HOMER's PAIRID_TO_NETWORK (Garin 21 anchor inheritance), then maps the
    HOMER name to the paper's naming via HOMER_NET_TO_PAPER_MOUSE.
    """
    idx_m = get_anchor_index(M_var)
    homer_net_ids = assign_networks(M_var, idx_m)  # int into NETWORKS
    homer_net_names = [NETWORKS[i] for i in homer_net_ids]
    paper_names = [HOMER_NET_TO_PAPER_MOUSE[n] for n in homer_net_names]

    # Same paper-aligned name set as human (extended with mouse-only Brainstem,
    # Frontoparietal which the paper does not include).
    if separate_aud:
        net_names = ["Visual", "Auditory", "SomatoMotor", "DorsAtten", "Salience",
                     "Limbic", "Control", "DMN", "Subcortical",
                     "HC_Limbic", "BF_Olfactory", "Frontoparietal", "Brainstem"]
    else:
        net_names = ["Visual", "SomatoMotor", "DorsAtten", "Salience",
                     "Limbic", "Control", "DMN", "Subcortical",
                     "HC_Limbic", "BF_Olfactory", "Frontoparietal", "Brainstem"]
    name_to_idx = {n: i for i, n in enumerate(net_names)}
    out = np.array([name_to_idx[p] for p in paper_names], dtype=np.int32)
    return out, net_names


def compute_network_mapping(pi: np.ndarray, mouse_net: np.ndarray, human_net: np.ndarray,
                             *, n_mouse: int, n_human: int) -> np.ndarray:
    """Aggregate π[m, h] over (mouse-net i, human-net j) → (n_mouse, n_human) matrix."""
    N = np.zeros((n_mouse, n_human), dtype=np.float64)
    for i in range(n_mouse):
        m_mask = (mouse_net == i)
        if not m_mask.any():
            continue
        for j in range(n_human):
            h_mask = (human_net == j)
            if not h_mask.any():
                continue
            N[i, j] = pi[np.ix_(m_mask, h_mask)].sum()
    return N


def score_mapping(N: np.ndarray, mouse_names: list[str], human_names: list[str],
                  *, target_pairs: list[tuple[str, str]]) -> dict:
    """Score diagonal-dominance for a set of mouse↔human network correspondences.

    For each target pair (mouse_net, human_net), check:
      - rank of human_net among all 8/9 human networks when ordered by N[mouse, :]
      - row-normalised mass on the target human-net
      - Z-score vs expected mass under the null where π is uniform on parcels.
    """
    # Sizes
    # Mouse uniform marginal: row-sum of N[i, :] is the fraction of mouse parcels
    # in mouse-net i, sums to 1 across mouse nets.
    row_sum = N.sum(axis=1)
    col_sum = N.sum(axis=0)
    total = N.sum()

    results = []
    for mouse_name, human_name in target_pairs:
        if mouse_name not in mouse_names or human_name not in human_names:
            results.append({
                "mouse_net": mouse_name, "human_net": human_name,
                "status": "missing"})
            continue
        i = mouse_names.index(mouse_name)
        j = human_names.index(human_name)
        if row_sum[i] == 0:
            results.append({
                "mouse_net": mouse_name, "human_net": human_name,
                "status": "empty_mouse_net"})
            continue
        row_norm = N[i, :] / row_sum[i]
        # Rank of target human-net (1 = top)
        order = (-N[i, :]).argsort()
        rank = int(np.where(order == j)[0][0]) + 1
        # Null expectation: if π was uniform-random across human parcels, each row
        # would route mass proportional to col_sum (human-net mass fraction).
        col_frac = col_sum / total if total > 0 else col_sum
        expected_frac = col_frac[j] * (row_sum.sum() / row_sum.sum())  # = col_frac[j]
        results.append({
            "mouse_net": mouse_name,
            "human_net": human_name,
            "rank": rank,
            "row_norm_mass": float(row_norm[j]),
            "expected_mass_null": float(expected_frac),
            "ratio_over_null": float(row_norm[j] / max(expected_frac, 1e-12)),
            "argmax_human_net": human_names[int(order[0])],
            "is_argmax_diagonal": bool(order[0] == j),
        })
    return {"per_pair": results}


def main():
    print("=" * 80)
    print("Pagani 2026 cross-species network validation")
    print("=" * 80)

    M, _ = load_cached("mouse", cache_dir="outputs/anndata")
    H, _ = load_cached("human", cache_dir="outputs/anndata")

    # Canonical coupling. This log is the BASELINE that experiments/whitesell_2021_dmn
    # compares against, so it must be on the same coupling as that experiment or the
    # comparison is not like-for-like. Print the sha256: a re-run proves nothing about
    # which input was used.
    pi = load_pi()
    prov = pi_provenance()
    print(f"\nLoading π: {prov['pi_file']}")
    print(f"  sha256: {prov['pi_sha256']}")
    print(f"  shape: {pi.shape}, sum: {pi.sum():.4f}")

    print("\nAssigning networks...")
    mouse_net, mouse_names = assign_mouse_paper_networks(M.var, separate_aud=True)
    human_net, human_names = assign_human_paper_networks(H.var, separate_aud=True)
    print(f"  mouse: {len(mouse_names)} networks, parcels = "
          f"{ {n: int((mouse_net==i).sum()) for i,n in enumerate(mouse_names)} }")
    print(f"  human: {len(human_names)} networks, parcels = "
          f"{ {n: int((human_net==i).sum()) for i,n in enumerate(human_names)} }")

    print("\nBuilding network-network mapping matrix from π...")
    N = compute_network_mapping(
        pi, mouse_net, human_net,
        n_mouse=len(mouse_names), n_human=len(human_names),
    )
    print(f"  N shape: {N.shape}, total mass: {N.sum():.4f}")

    # Print row-normalised matrix
    print("\nRow-normalised network mapping (mouse network → human network mass %):")
    row_sums = N.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    N_rn = N / row_sums
    # Print header
    print(f"  {'mouse_net':<18s} | " + " ".join(f"{h[:6]:>6s}" for h in human_names))
    for i, m in enumerate(mouse_names):
        if N[i].sum() == 0:
            continue
        cells = " ".join(f"{N_rn[i, j]*100:6.1f}" for j in range(len(human_names)))
        argmax_marker = " ←★" if int(N_rn[i].argmax()) < len(human_names) else ""
        argmax_name = human_names[int(N_rn[i].argmax())]
        print(f"  {m:<18s} | {cells}   (argmax: {argmax_name})")

    # Score the canonical name-based correspondences the paper relies on
    target_pairs = [
        ("Visual", "Visual"),
        ("Auditory", "Auditory"),
        ("SomatoMotor", "SomatoMotor"),
        ("DMN", "DMN"),
        ("Salience", "Salience"),
        ("HC_Limbic", "Limbic"),
        ("Subcortical", "Subcortical"),
        ("BF_Olfactory", "Subcortical"),  # mouse BF/olfactory ≈ human subcortical or limbic
    ]
    print("\nDiagonal-dominance scoring (target pairs from the paper):")
    score = score_mapping(N, mouse_names, human_names, target_pairs=target_pairs)
    for r in score["per_pair"]:
        if r.get("status"):
            print(f"  {r['mouse_net']:<14s} → {r['human_net']:<14s} : {r['status']}")
        else:
            star = "★" if r["is_argmax_diagonal"] else " "
            print(f"  {r['mouse_net']:<14s} → {r['human_net']:<14s} : "
                  f"{star} rank={r['rank']}, mass={r['row_norm_mass']*100:5.1f}% "
                  f"(null={r['expected_mass_null']*100:4.1f}%, "
                  f"×{r['ratio_over_null']:4.1f})")

    # Summary statistics
    is_diag = [r["is_argmax_diagonal"] for r in score["per_pair"] if "is_argmax_diagonal" in r]
    n_pairs_scorable = len(is_diag)
    n_diag_argmax = sum(is_diag)
    mean_ratio = float(np.mean([r["ratio_over_null"] for r in score["per_pair"]
                                if "ratio_over_null" in r]))
    print(f"\nSummary:")
    print(f"  pairs scored:         {n_pairs_scorable}")
    print(f"  diagonal is argmax:   {n_diag_argmax}/{n_pairs_scorable}")
    print(f"  mean ratio over null: {mean_ratio:.2f}×")

    out = {
        **prov,
        "mouse_networks": mouse_names,
        "human_networks": human_names,
        "mouse_net_sizes": {n: int((mouse_net == i).sum()) for i, n in enumerate(mouse_names)},
        "human_net_sizes": {n: int((human_net == i).sum()) for i, n in enumerate(human_names)},
        "mapping_matrix": N.tolist(),
        "row_normalised_matrix": N_rn.tolist(),
        "target_pair_scores": score["per_pair"],
        "summary": {
            "n_pairs_scored": n_pairs_scorable,
            "n_diagonal_argmax": n_diag_argmax,
            "fraction_diagonal_argmax": n_diag_argmax / max(n_pairs_scorable, 1),
            "mean_ratio_over_null": mean_ratio,
        },
    }
    out_path = Path("outputs/logs/autism_subtypes_network_crossval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
