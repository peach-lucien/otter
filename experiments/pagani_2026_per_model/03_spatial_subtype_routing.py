"""Direction 1 — parcel-resolution spatial subtype routing through π.

The earlier subtype translation (Test 2c / 01_per_model_clustering.py) drives the
mouse side from Pagani's coarse 9-network matrices. Here we use the genuinely
spatial **Fig 1d occurrence maps** instead: for each subtype, the per-voxel count
(0–5) of how many models show a consistent hyper/hypo effect.

Bridge (verified, no fragile voxel transform):
  • The occurrence maps and the 13 conserved-region masks are co-registered
    (identical affine), so we read the mean occurrence within each region mask
    directly, in their own space.
  • HOMER's 1,864 mouse parcels carry Allen region-vote names, which map to the
    13 conserved regions by keyword (verified anatomically: thalamus→thalamic
    nuclei, caudoputamen→Caudoputamen/Striatum, etc.).

Pipeline: occurrence map → mean per conserved region → assign to mouse parcels →
route through π → human-parcel prediction → aggregate to 8 Pagani human networks
→ compare to observed human subtype pattern (Fig 4e), with a permuted-π null.

Usage:
    PYTHONPATH=src python experiments/pagani_2026_per_model/03_spatial_subtype_routing.py
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "autism_subtypes"))

from homer.data import DATA_DIR, load_cached  # noqa: E402

ncv = import_module("01_network_crossvalidation")
st = import_module("04_subtype_translation")

PAGANI = Path(DATA_DIR) / "pagani"
OCC = {
    "hyper": PAGANI / "cluster1_AMBA_occurrence_map_pos_cohens_d_0.8.nii.gz",
    "hypo":  PAGANI / "cluster2_AMBA_occurrence_map_neg_cohens_d_0.8.nii.gz",
}
MASKDIR = PAGANI / "Region_masks"

# Allen region-vote keyword rules → 13 conserved regions (verified bridge).
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
REGIONS = list(RULES)


def parcel_to_region(var) -> np.ndarray:
    rv = [str(x).lower() for x in var["region_vote_ns_aba"].values]
    out = np.array(["(none)"] * len(rv), dtype=object)
    for i, n in enumerate(rv):
        for region, keys in RULES.items():
            if any(k in n for k in keys):
                out[i] = region
                break
    return out


def region_occurrence(subtype: str) -> dict[str, float]:
    occ = nib.load(str(OCC[subtype])).get_fdata()
    vals = {}
    for r in REGIONS:
        mpath = MASKDIR / f"{r}.nii.gz"
        if not mpath.exists():
            vals[r] = 0.0
            continue
        mask = nib.load(str(mpath)).get_fdata() > 0
        v = occ[mask]
        vals[r] = float(v[v > 0].mean()) if (v > 0).any() else 0.0
    return vals


def build_mouse_vector(region_val: dict[str, float], assign: np.ndarray) -> np.ndarray:
    vec = np.zeros(len(assign), dtype=float)
    for r in REGIONS:
        vec[assign == r] = region_val[r]
    return vec


def human_aggregator(H):
    hpn, hnames = ncv.assign_human_paper_networks(H.var, separate_aud=True)
    a, s = hnames.index("Auditory"), hnames.index("SomatoMotor")
    hpn = hpn.copy()
    hpn[hpn == a] = s
    nets = st.PAGANI_HUMAN_NETS

    def agg(values):
        out = np.zeros(len(nets))
        for i, n in enumerate(nets):
            if n in hnames:
                mask = hpn == hnames.index(n)
                out[i] = values[mask].mean() if mask.any() else 0.0
        return out
    return agg, nets


def main():
    print("=" * 78)
    print("Direction 1 — parcel-resolution spatial subtype routing (occurrence maps)")
    print("=" * 78)

    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = np.load(ROOT / "outputs/coupling/pi_fc_plus_SC.npy")
    assign = parcel_to_region(M.var)
    print(f"\nmouse parcels matched to conserved regions: "
          f"{int((assign != '(none)').sum())}/{len(assign)}")

    region_val = {s: region_occurrence(s) for s in ("hyper", "hypo")}
    print("\nmean occurrence per conserved region (hyper | hypo):")
    for r in REGIONS:
        print(f"  {r:<22} {region_val['hyper'][r]:>5.2f} | {region_val['hypo'][r]:>5.2f}")

    agg, nets = human_aggregator(H)

    # Route each subtype's spatial signature through π.
    pred = {}
    for s in ("hyper", "hypo"):
        mvec = build_mouse_vector(region_val[s], assign)
        pred[s] = agg(mvec @ pi)

    # Observed human subtype network pattern (Fig 4e).
    data = st.load_pagani_subtype_matrices()
    obs = {s: np.array([
        {n: float(v) for n, v in zip(nets, st.network_intensity(data[f"human_{s}"], "abs_rowcol_sum"))}[n]
        for n in nets]) for s in ("hyper", "hypo")}

    # Contrast test (hyper − hypo), which cancels human-network size effects.
    pred_delta = pred["hyper"] - pred["hypo"]
    obs_delta = obs["hyper"] - obs["hypo"]
    r_p, p_p = pearsonr(pred_delta, obs_delta)
    r_s, p_s = spearmanr(pred_delta, obs_delta)

    # Subtype-specificity cross-correlation.
    xcorr = {f"pred_{a}__obs_{b}": float(np.corrcoef(pred[a], obs[b])[0, 1])
             for a in ("hyper", "hypo") for b in ("hyper", "hypo")}

    # Permuted-π null on the contrast.
    rng = np.random.default_rng(0)
    null = []
    for _ in range(500):
        perm = rng.permutation(pi.shape[0])
        pn_hyper = agg(build_mouse_vector(region_val["hyper"], assign) @ pi[perm])
        pn_hypo = agg(build_mouse_vector(region_val["hypo"], assign) @ pi[perm])
        null.append(pearsonr(pn_hyper - pn_hypo, obs_delta)[0])
    null = np.array(null)
    emp_p = float((null >= r_p).mean())

    print(f"\nContrast (hyper−hypo) predicted vs observed:")
    print(f"  Pearson r = {r_p:+.3f} (analytical p={p_p:.3f}); "
          f"Spearman ρ = {r_s:+.3f}")
    print(f"  permuted-π null mean={null.mean():+.3f} "
          f"(95% CI {np.percentile(null,2.5):+.3f}..{np.percentile(null,97.5):+.3f}); "
          f"empirical p = {emp_p:.3f}")
    print(f"\nSubtype-specificity cross-correlation:")
    for k, v in xcorr.items():
        print(f"  {k:<24} {v:+.3f}")

    out = {
        "n_matched_parcels": int((assign != "(none)").sum()),
        "region_occurrence": region_val,
        "human_networks": nets,
        "predicted": {s: pred[s].tolist() for s in pred},
        "observed": {s: obs[s].tolist() for s in obs},
        "contrast_pearson_r": float(r_p),
        "contrast_spearman_r": float(r_s),
        "contrast_empirical_p": emp_p,
        "null_mean": float(null.mean()),
        "cross_correlation": xcorr,
    }
    out_path = ROOT / "outputs" / "logs" / "pagani_spatial_subtype_routing.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
