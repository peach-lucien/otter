"""HOMER × Whitesell 2021 DMN refinement.

[Whitesell et al. 2021, Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(21)00006-X)
characterised the mouse default mode network at high resolution using Allen
Mouse Brain Connectivity (MBC) tract-tracing + rsfMRI, providing the most
careful published mouse-DMN boundary. Their cross-species comparison maps the
mouse DMN onto human Yeo-7 DMN regions.

This experiment tests whether HOMER's DMN-DMN correspondence (currently 41 %
row-mass per Pagani Test 1) sharpens when we use Whitesell's more careful
mouse-DMN parcel definition rather than HOMER's PAIRID-derived 'frontal_dmn'
+ 'temporal_dmn' networks.

Mouse-DMN regions per Whitesell 2021 (DSURQE labels):
  - mPFC: Prelimbic area, Infralimbic area
  - ACC:  Anterior cingulate area (24a/24b dorsal + ventral)
  - RSC:  Retrosplenial area (29a, 29b, 29c, dorsal part)
  - PPC:  Posterior parietal association areas
  - Hippocampal formation: dorsal CA1, Subiculum
  - Entorhinal cortex, medial part

Test:
  1. Identify HOMER mouse parcels with DSURQE labels in these regions.
  2. Aggregate π for these parcels → human-side mass distribution (2,094-vec).
  3. Aggregate to Yeo-7 networks → row-mass per human network.
  4. Compare DMN row-mass against the baseline (Pagani Test 1, 41 %).

If Whitesell-DMN produces a higher DMN row-mass than HOMER's PAIRID-derived
DMN, that's evidence the current PAIRID scheme underspecifies the mouse DMN
(missing PPC, RSC, dorsal hippocampus), and an opportunity for a
whitesell_dmn anchor pack.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "autism_subtypes"))

from homer.data import load_cached
from homer.data.anchor_packs._dsurqe import mouse_parcels_in_dsurqe_region


# Whitesell 2021 mouse DMN: union of these DSURQE regions
WHITESELL_DMN_REGIONS = [
    # Medial prefrontal, core DMN node
    "Prelimbic area",
    "Infralimbic area",
    # Anterior cingulate, core DMN node
    "Anterior cingulate area",
    # Retrosplenial, core posterior DMN node (analog of human PCC)
    "Retrosplenial area",
    # Posterior parietal. DMN posterior hub
    "Posterior parietal association areas",
    # Hippocampal formation (dorsal hippocampus + subiculum)
    "Field CA1",
    "Subiculum",
    # Medial entorhinal. DMN-adjacent
    "Entorhinal area, medial part, dorsal zone",
    "Entorhinal area, medial part, ventral zone",
]


def main():
    print("=" * 80)
    print("HOMER × Whitesell 2021 DMN refinement")
    print("=" * 80)

    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = np.load(ROOT / "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")
    print(f"  π: {pi.shape}, total mass {pi.sum():.4f}")

    # Identify Whitesell DMN mouse parcels via DSURQE
    print("\nResolving Whitesell 2021 DMN regions via DSURQE labels:")
    whitesell_dmn_parcels = set()
    per_region = {}
    for region in WHITESELL_DMN_REGIONS:
        try:
            idxs = mouse_parcels_in_dsurqe_region(M.var, region, atlas_root=ROOT)
            per_region[region] = len(idxs)
            whitesell_dmn_parcels.update(idxs)
            print(f"  {region:<48s}: {len(idxs):>3d} parcels")
        except KeyError as e:
            print(f"  {region:<48s}: NOT FOUND  ({e})")
    whitesell_dmn_parcels = sorted(whitesell_dmn_parcels)
    n_dmn = len(whitesell_dmn_parcels)
    print(f"\n  Whitesell-DMN mouse parcels (union): {n_dmn}/{len(M.var)} "
          f"({100*n_dmn/len(M.var):.1f}%)")

    # Human network assignment (Yeo-7 + Subcortical, audit-corrected)
    from importlib import import_module
    nc = import_module("01_network_crossvalidation")
    human_net, human_paper_names = nc.assign_human_paper_networks(H.var, separate_aud=True)
    aud_idx = human_paper_names.index("Auditory")
    som_idx = human_paper_names.index("SomatoMotor")
    human_net = human_net.copy()
    human_net[human_net == aud_idx] = som_idx
    pagani_human = ["Control", "DMN", "DorsAtten", "Limbic", "Salience",
                    "SomatoMotor", "Visual", "Subcortical"]
    h_name_to_idx = {n: human_paper_names.index(n) for n in pagani_human}

    # Aggregate π for the Whitesell DMN parcels → human-side mass
    whitesell_pi_row = pi[whitesell_dmn_parcels].sum(axis=0)   # (2094,)
    # Row-normalize
    row_total = whitesell_pi_row.sum()
    whitesell_pi_norm = whitesell_pi_row / row_total

    # Aggregate to Yeo-7 networks
    print(f"\nWhitesell-DMN → human Yeo-7 row-mass:")
    print(f"{'Network':<14s} | {'row-mass':>10s} | {'n_h_parcels':>12s} | {'expected null':>14s} | {'ratio':>7s}")
    print("-" * 75)
    n_h = pi.shape[1]
    results = []
    for net in pagani_human:
        idx = h_name_to_idx[net]
        mask = human_net == idx
        mass = float(whitesell_pi_norm[mask].sum())
        n_h_parc = int(mask.sum())
        expected = n_h_parc / n_h
        ratio = mass / expected if expected > 0 else float("nan")
        marker = "★" if net == "DMN" else " "
        print(f"  {net:<12s} {marker} | {mass*100:>8.1f}% | {n_h_parc:>12d} | "
              f"{expected*100:>12.1f}% | {ratio:>6.2f}×")
        results.append({"network": net, "row_mass": mass,
                         "n_human_parcels": n_h_parc,
                         "expected_null": expected, "ratio_over_null": ratio,
                         "is_dmn": net == "DMN"})

    dmn_result = next(r for r in results if r["is_dmn"])

    # Compare to Pagani Test 1 (PAIRID-derived DMN → human DMN)
    print(f"\nComparison to existing tests:")
    pagani_result = json.loads((ROOT / "outputs/logs/autism_subtypes_network_crossval.json").read_text())
    # Extract DMN→DMN from Pagani Test 1
    for p in pagani_result["target_pair_scores"]:
        if p.get("mouse_net") == "DMN" and p.get("human_net") == "DMN":
            pagani_dmn = p["row_norm_mass"]
            break
    else:
        pagani_dmn = None
    coletta_result = json.loads((ROOT / "outputs/logs/coletta_2020_cross_species_rsn.json").read_text())
    # Coletta has frontal_dmn → DMN and temporal_dmn → DMN
    coletta_frontal = next((p["row_norm_mass"] for p in coletta_result["sub_test_A_labeled_correspondence"]["per_pair_scores"]
                           if p.get("mouse_net") == "frontal_dmn" and p.get("human_net") == "DMN"), None)
    coletta_temporal = next((p["row_norm_mass"] for p in coletta_result["sub_test_A_labeled_correspondence"]["per_pair_scores"]
                            if p.get("mouse_net") == "temporal_dmn" and p.get("human_net") == "DMN"), None)
    print(f"  Pagani Test 1 (HOMER 'DMN' net → human DMN):       "
          f"{pagani_dmn*100:.1f}%" if pagani_dmn else "  Pagani Test 1: N/A")
    print(f"  Coletta sub-test A (frontal_dmn → DMN):            "
          f"{coletta_frontal*100:.1f}%" if coletta_frontal else "  Coletta: N/A")
    print(f"  Coletta sub-test A (temporal_dmn → DMN):           "
          f"{coletta_temporal*100:.1f}%" if coletta_temporal else "  Coletta: N/A")
    print(f"  Whitesell 2021 DMN (this experiment):              "
          f"{dmn_result['row_mass']*100:.1f}%  [{dmn_result['ratio_over_null']:.2f}× null]")

    if dmn_result['row_mass'] > (pagani_dmn or 0.41):
        verdict = (f"Whitesell DMN ({dmn_result['row_mass']*100:.1f}%) BEATS Pagani's PAIRID-derived "
                   f"DMN ({(pagani_dmn or 0)*100:.1f}%) → refining the mouse-DMN parcel set sharpens HOMER's DMN→DMN correspondence")
    else:
        verdict = (f"Whitesell DMN ({dmn_result['row_mass']*100:.1f}%) is in the same range as Pagani's PAIRID-derived DMN "
                   f"→ HOMER's existing DMN definition is consistent with Whitesell's careful definition")
    print(f"\nVerdict: {verdict}")

    # Save
    out = {
        "n_whitesell_dmn_parcels": int(n_dmn),
        "fraction_of_brain":      n_dmn / len(M.var),
        "per_region_counts":      per_region,
        "per_yeo7_network":       results,
        "comparison": {
            "pagani_DMN_to_DMN":         pagani_dmn,
            "coletta_frontal_dmn_to_DMN": coletta_frontal,
            "coletta_temporal_dmn_to_DMN": coletta_temporal,
            "whitesell_DMN_to_DMN":      dmn_result["row_mass"],
        },
        "verdict": verdict,
    }
    out_path = ROOT / "outputs" / "logs" / "whitesell_2021_dmn_refinement.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
