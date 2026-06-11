"""Per-mouse-model subtype translation through HOMER's π  (CORRECTED, 2026-06-10).

This supersedes the earlier version of this script, which relied on a
biological-prior guess for the hyper/hypo subtype of each model — a prior that
turned out to be **inverted** (it labelled Fmr1/Chd8/Tsc2 as hypo when they are
hyper). See `DATA_VALIDATION_2026-06-10.md`.

What changed now that the Gozzi lab shared the clean data
(`data_crossspecies/pagani/`):

  1. The clean Fig 1c matrix `sorted_etiology_by_feature_matrix.csv`
     (20 models × 1,491 voxelwise weighted-degree-centrality features) replaces
     the Excel-corrupted MOESM6 load — no more outlier masking.

  2. The per-model hyper/hypo labels are no longer guessed. The CSV is *sorted*
     by Pagani's hierarchical clustering, so the subtype split falls exactly on
     row order: rows 1–9 = hyperconnectivity (n=9), rows 10–20 =
     hypoconnectivity (n=11). We *verify* this from the data itself (mean global
     connectivity > 0 for hyper, < 0 for hypo) rather than asserting it.

The translation itself (mouse subtype signature → human-parcel prediction via π)
reuses the validated Test 2 machinery in `04_subtype_translation.py`. It does NOT
depend on decoding the 1,491 features to voxels (which is not robustly possible —
see the validation note); the per-subtype network signatures come from Pagani's
own ED Fig 1 / Fig 4e network matrices.

Outputs:
  - outputs/logs/pagani_subtype_translation_corrected.json
  - outputs/figures/pagani_subtype_translation_corrected.png   (via 02_plot.py)

Usage:
    PYTHONPATH=src python experiments/pagani_2026_per_model/01_per_model_clustering.py
"""
from __future__ import annotations

import csv
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "autism_subtypes"))

from homer.data import DATA_DIR, load_cached  # noqa: E402

st = import_module("04_subtype_translation")
ncv = import_module("01_network_crossvalidation")

CLEAN_CSV = Path(DATA_DIR) / "pagani" / "sorted_etiology_by_feature_matrix.csv"
PI_PATH = ROOT / "outputs" / "coupling" / "pi_fc_plus_SC_with_all_packs.npy"

# Pagani's hierarchical-clustering row split (Fig 1c): first 9 rows are the
# hyperconnectivity subtype, the remaining 11 the hypoconnectivity subtype.
N_HYPER = 9  # rows 0..8


# ---------------------------------------------------------------------------
# 1. Load clean Fig 1c and derive + verify subtype labels
# ---------------------------------------------------------------------------
def load_clean_figura_1c() -> tuple[np.ndarray, list[str]]:
    labels, rows = [], []
    with open(CLEAN_CSV) as fh:
        for r in csv.reader(fh):
            labels.append(r[0])
            rows.append([float(x) for x in r[1:]])
    return np.asarray(rows, dtype=np.float64), labels


def derive_and_verify_subtypes(X: np.ndarray, labels: list[str]) -> list[str]:
    """Assign subtype by row order, then verify against the data (mean sign)."""
    subtype = ["hyper" if i < N_HYPER else "hypo" for i in range(len(labels))]
    row_mean = X.mean(axis=1)

    # Verification: hyper rows should have positive mean global connectivity,
    # hypo rows negative. This is what makes the row-order split trustworthy.
    hyper_means = row_mean[:N_HYPER]
    hypo_means = row_mean[N_HYPER:]
    assert (hyper_means > 0).all(), (
        "Expected all hyper rows to have positive mean connectivity; "
        f"got {hyper_means}")
    assert (hypo_means < 0).all(), (
        "Expected all hypo rows to have negative mean connectivity; "
        f"got {hypo_means}")
    return subtype


# ---------------------------------------------------------------------------
# 2. Leave-one-out per-model membership on the hyper↔hypo axis
# ---------------------------------------------------------------------------
def loo_membership(X: np.ndarray, subtype: list[str]) -> list[dict]:
    """For each model, correlate its 1,491-feature vector with the mean
    signature of each subtype — *excluding the model itself* from the mean, so
    the placement isn't circular. Returns hyper/hypo correlations and a signed
    membership score (hyper minus hypo correlation)."""
    sub = np.array(subtype)
    hyper_idx = np.where(sub == "hyper")[0]
    hypo_idx = np.where(sub == "hypo")[0]
    out = []
    for i in range(X.shape[0]):
        hy = [j for j in hyper_idx if j != i]
        ho = [j for j in hypo_idx if j != i]
        hyper_sig = X[hy].mean(axis=0)
        hypo_sig = X[ho].mean(axis=0)
        r_hyper = float(np.corrcoef(X[i], hyper_sig)[0, 1])
        r_hypo = float(np.corrcoef(X[i], hypo_sig)[0, 1])
        out.append({
            "r_to_hyper_signature": r_hyper,
            "r_to_hypo_signature": r_hypo,
            "membership_score": r_hyper - r_hypo,   # >0 ⇒ closer to hyper
            "predicted_side": "hyper" if r_hyper > r_hypo else "hypo",
        })
    return out


# ---------------------------------------------------------------------------
# 3. Subtype signature → human-parcel prediction via π  (reuses Test 2)
# ---------------------------------------------------------------------------
def subtype_translation_through_pi() -> dict:
    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = np.load(PI_PATH)

    mpn, mnames = ncv.assign_mouse_paper_networks(M.var, separate_aud=True)
    hpn, hnames = ncv.assign_human_paper_networks(H.var, separate_aud=True)
    # Merge our human Auditory into SomatoMotor for the 8-net Pagani comparison.
    a, s = hnames.index("Auditory"), hnames.index("SomatoMotor")
    hpn_m = hpn.copy()
    hpn_m[hpn == a] = s

    data = st.load_pagani_subtype_matrices()
    metric = "abs_rowcol_sum"

    preds, obs = {}, {}
    for subtype in ("hyper", "hypo"):
        mouse_M = data[f"mouse_{subtype}"]
        human_M = data[f"human_{subtype}"]
        m_int = st.network_intensity(mouse_M, metric=metric)
        m_dict = {st.PAGANI_MOUSE_NETS[i]: float(m_int[i])
                  for i in range(len(st.PAGANI_MOUSE_NETS))}
        parcel_vals = st.mouse_intensity_to_parcel_values(mpn, mnames, m_dict)
        human_pred = parcel_vals @ pi  # (2094,)
        pred_net = st.aggregate_human_parcels_to_networks(
            human_pred, hpn_m, hnames, st.PAGANI_HUMAN_NETS)
        h_int = st.network_intensity(human_M, metric=metric)
        obs_net = {st.PAGANI_HUMAN_NETS[i]: float(h_int[i])
                   for i in range(len(st.PAGANI_HUMAN_NETS))}
        preds[subtype] = pred_net
        obs[subtype] = obs_net

    # Cross correlation matrix between predicted and observed subtype patterns.
    nets = st.PAGANI_HUMAN_NETS
    def vec(d):
        return np.array([d[n] for n in nets], dtype=float)
    xcorr = {}
    for ps in ("hyper", "hypo"):
        for os_ in ("hyper", "hypo"):
            pv, ov = vec(preds[ps]), vec(obs[os_])
            xcorr[f"pred_{ps}__obs_{os_}"] = float(np.corrcoef(pv, ov)[0, 1])

    specificity_hyper = xcorr["pred_hyper__obs_hyper"] > xcorr["pred_hyper__obs_hypo"]
    specificity_hypo = xcorr["pred_hypo__obs_hypo"] > xcorr["pred_hypo__obs_hyper"]
    return {
        "human_networks": nets,
        "predicted": preds,
        "observed": obs,
        "cross_correlation": xcorr,
        "subtype_specific_hyper": bool(specificity_hyper),
        "subtype_specific_hypo": bool(specificity_hypo),
    }


def main():
    print("=" * 78)
    print("Pagani 2026 — corrected per-model subtype translation through π")
    print("=" * 78)

    X, labels = load_clean_figura_1c()
    print(f"\nClean Fig 1c: {X.shape[0]} models × {X.shape[1]} features "
          f"(range [{X.min():.2f}, {X.max():.2f}], no outlier masking)")

    subtype = derive_and_verify_subtypes(X, labels)
    n_hyper = subtype.count("hyper")
    print(f"\nSubtype split (verified by mean-connectivity sign): "
          f"{n_hyper} hyper / {len(subtype) - n_hyper} hypo")
    print(f"\n{'row':>3}  {'model':<14} {'mean_conn':>9}  {'subtype':>7}")
    rmean = X.mean(axis=1)
    for i, lbl in enumerate(labels):
        print(f"{i + 1:>3}  {lbl:<14} {rmean[i]:>9.3f}  {subtype[i]:>7}")

    members = loo_membership(X, subtype)
    n_consistent = sum(members[i]["predicted_side"] == subtype[i]
                       for i in range(len(labels)))
    print(f"\nLeave-one-out membership consistency: "
          f"{n_consistent}/{len(labels)} models fall on their own subtype side")

    print("\nRouting per-subtype signatures through π → human space ...")
    trans = subtype_translation_through_pi()
    xc = trans["cross_correlation"]
    print("  cross-species correlation (predicted human ↔ observed human):")
    print(f"    pred_hyper · obs_hyper = {xc['pred_hyper__obs_hyper']:+.3f}  "
          f"(vs obs_hypo {xc['pred_hyper__obs_hypo']:+.3f})")
    print(f"    pred_hypo  · obs_hypo  = {xc['pred_hypo__obs_hypo']:+.3f}  "
          f"(vs obs_hyper {xc['pred_hypo__obs_hyper']:+.3f})")
    print(f"  subtype-specific (hyper): {trans['subtype_specific_hyper']}")
    print(f"  subtype-specific (hypo):  {trans['subtype_specific_hypo']}")

    out = {
        "n_models": len(labels),
        "n_hyper": n_hyper,
        "n_hypo": len(labels) - n_hyper,
        "models": [
            {"row": i + 1, "model": labels[i], "subtype": subtype[i],
             "mean_connectivity": float(rmean[i]), **members[i]}
            for i in range(len(labels))
        ],
        "loo_consistency": f"{n_consistent}/{len(labels)}",
        "translation": trans,
        "note": ("Supersedes inverted-prior version. Subtype labels verified "
                 "from data; translation independent of 1,491-feature decode."),
    }
    out_path = ROOT / "outputs" / "logs" / "pagani_subtype_translation_corrected.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
