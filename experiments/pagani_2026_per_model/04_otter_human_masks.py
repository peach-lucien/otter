"""OTTER-derived human subtype masks, replace Pagani's name-matched bridge with π.

Pagani 2026 build the human hypo/hyper subtype masks by NAME: they pick the mouse
"prominent" dysconnectivity regions (hypo = anterior+middle cingulate, insula,
motor cortex, striatum; hyper = amygdala, hippocampus, striatum. Methods, Supp
Fig 2b) and then score the *same-named* human regions. That mouse→human step is
exactly what OTTER's learned coupling π replaces.

Steps:
  1. Mark the mouse parcels in Pagani's 5 hypo-prominent / 3 hyper-prominent
     conserved regions (Allen region-name bridge).
  2. Route each prominent region through π (transport-weighted) to give the human
     Yeo-7/subcortical networks OTTER sends it to, the region-by-region homology
     test against Pagani's name assumption.
  3. Aggregate → data-driven human hypo/hyper coupling maps → thresholded masks,
     saved for the ABIDE re-subtyping step (05_*).

This uses discrete region/network correspondence, which survives spin nulls,
rather than the continuous-map correlation, which does not.

Usage:
    PYTHONPATH=src python experiments/pagani_2026_per_model/04_otter_human_masks.py
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "autism_subtypes"))

from otter.data import load_cached, load_pi  # noqa: E402

ncv = import_module("01_network_crossvalidation")


# Allen region-vote → Pagani 13 conserved regions.
RULES = {
    "amygdala":             ["amygdal"],
    "auditory_cortex":      ["auditory area"],
    "caudoputamen":         ["caudoputamen", "striatum"],
    "cingulate_anterior":   ["anterior cingulate"],
    "cingulate_middle":     ["cingulate area dorsal", "midcingulate"],
    "hippocampus":          ["field ca", "dentate gyrus", "hippocamp", "subiculum"],
    "hypothalamus":         ["hypothalam"],
    "insula":               ["agranular insular", "insular area", "visceral area"],
    "motor_cortex":         ["primary motor", "secondary motor", "motor area"],
    "retrosplenial_cortex": ["retrosplenial"],
    "somatosensory_cortex": ["somatosensory"],
    "thalamus":             ["thalam", "geniculate", "ventral post"],
    "visual_cortex":        ["visual area", "primary visual"],
}

# Pagani's prominent dysconnectivity regions (Methods; striatum == caudoputamen).
HYPO_PROMINENT = ["cingulate_anterior", "cingulate_middle", "insula",
                  "motor_cortex", "caudoputamen"]
HYPER_PROMINENT = ["amygdala", "hippocampus", "caudoputamen"]

# What name-matching would PREDICT each mouse region maps to on the human side
# (Yeo-7 + Subcortical), to score whether π agrees with the by-name assumption.
NAME_EXPECTED = {
    "cingulate_anterior": {"DMN", "Control", "Salience"},
    "cingulate_middle":   {"DMN", "Control", "Salience"},
    "insula":             {"Salience"},
    "motor_cortex":       {"SomatoMotor"},
    "caudoputamen":       {"Subcortical"},
    "amygdala":           {"Subcortical"},
    "hippocampus":        {"Subcortical"},
}


def parcel_to_region(var):
    rv = [str(x).lower() for x in var["region_vote_ns_aba"].values]
    out = np.array(["(none)"] * len(rv), dtype=object)
    for i, n in enumerate(rv):
        for region, keys in RULES.items():
            if any(k in n for k in keys):
                out[i] = region
                break
    return out


def route(mouse_ind, pi):
    den = pi.sum(0)
    out = np.full(pi.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = (mouse_ind @ pi)[ok] / den[ok]
    return out


def main():
    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = load_pi().astype(np.float64)   # canonical coupling
    assign = parcel_to_region(M.var)
    hnet, hnames = ncv.assign_human_paper_networks(H.var, separate_aud=True)

    print("=" * 78)
    print("Region-by-region: where does π send each mouse prominent region?")
    print("(does OTTER agree with Pagani's name-matched human homologue?)")
    print("=" * 78)
    agree = 0
    region_rows = []
    for region in HYPO_PROMINENT + HYPER_PROMINENT:
        ind = (assign == region).astype(float)
        if ind.sum() == 0:
            print(f"  {region:<22} : no mouse parcels"); continue
        hcoup = route(ind, pi)
        # aggregate human coupling by Yeo network
        net_mass = {}
        for j, net in enumerate(hnames):
            m = (hnet == j) & np.isfinite(hcoup)
            net_mass[net] = float(np.nansum(hcoup[m]))
        top = sorted(net_mass, key=net_mass.get, reverse=True)[:3]
        exp = NAME_EXPECTED.get(region, set())
        hit = "✓" if (set(top[:1]) & exp or (region == "cingulate_middle" and top[0] in exp)) else ("~" if set(top[:3]) & exp else "✗")
        if set(top[:1]) & exp:
            agree += 1
        print(f"  {region:<22} → top human nets: {', '.join(top)}   "
              f"(name-expected: {', '.join(sorted(exp)) or '?'})  {hit}")
        region_rows.append({"mouse_region": region, "top_human_nets": top,
                             "name_expected": sorted(exp), "argmax_agrees": bool(set(top[:1]) & exp)})
    print(f"\nπ argmax agrees with name-matched homologue: {agree}/{len(region_rows)} regions")

    # ---- aggregate hypo / hyper human masks (data-driven) ----
    def build_mask(regions, pct=80):
        ind = np.isin(assign, regions).astype(float)
        coup = route(ind, pi)
        thr = np.nanpercentile(coup, pct)
        mask = (coup >= thr) & np.isfinite(coup)
        # Yeo composition of the mask
        comp = {}
        for j, net in enumerate(hnames):
            comp[net] = int((mask & (hnet == j)).sum())
        return coup, mask, comp, float(thr)

    out = {"region_routing": region_rows, "argmax_agree": agree,
           "n_regions": len(region_rows), "masks": {}}
    print(f"\n{'='*78}\nData-driven human masks (top-20% coupled human parcels)\n{'='*78}")
    for label, regions in [("hypo", HYPO_PROMINENT), ("hyper", HYPER_PROMINENT)]:
        coup, mask, comp, thr = build_mask(regions)
        top_nets = sorted(comp, key=comp.get, reverse=True)[:4]
        print(f"  {label:<5} mask: {int(mask.sum())} human parcels; "
              f"top Yeo nets: {', '.join(f'{n}({comp[n]})' for n in top_nets)}")
        out["masks"][label] = {
            "n_parcels": int(mask.sum()),
            "parcel_indices": np.where(mask)[0].tolist(),
            "yeo_composition": comp,
            "coupling_threshold_pct80": thr,
            "coupling_vector": np.where(np.isfinite(coup), coup, 0.0).tolist(),
        }
    out_path = ROOT / "outputs" / "logs" / "pagani_otter_human_masks.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}  (feeds 05_abide_otter_subtyping.py)")


if __name__ == "__main__":
    main()
