#!/usr/bin/env python3
"""Score the Hodge individual layer markers on the SAME footing as the layer contrasts.

WHY THIS EXISTS
---------------
Section 3 contrasts two numbers:

    individual markers   mean r = 0.23,  6 of 7 significant   (01_layer_marker_validation.py)
    layer contrasts      mean r = 0.07,  3 of 4 non-significant (03_areal_type_reframe.py)

and reads the gap as evidence that π carries areal but not laminar organisation. The two numbers
are not comparable, in two independent ways:

  1. MASK.  01 correlates over all 2,094 human parcels. 03 restricts to the 1,768 cortical parcels
     of Schaefer-400. Layer contrasts are only defined in cortex, so the marker number includes
     subcortical parcels the contrast number cannot.

  2. NULL.  01 uses a permuted-π null (200 shuffles of π's rows), which destroys all spatial
     structure and is easy to beat. 03 uses a translation spin null (1,000 rotations of the mouse
     input routed through the real π), which preserves the mouse map's spatial autocorrelation.
     "6 of 7 significant" and "3 of 4 non-significant" are therefore counts against different and
     non-comparable nulls.

Section 3 also states that spin nulls were used "throughout", which is not true of the per-marker
test as it stands.

This script re-scores the seven markers cortex-only against the translation spin null, so the two
halves of the areal-versus-laminar claim are measured the same way. It reports the original
whole-brain values alongside, so the size of the change is visible rather than silent. It does not
modify either existing script or their logs.

Usage:
    cd homer && PYTHONPATH=src python experiments/hodge_2019_cortical_layers/04_marker_like_for_like.py

Writes: outputs/logs/hodge_markers_like_for_like.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached, load_pi, pi_provenance                    # noqa: E402
from homer.data.atlas_regions import (                                        # noqa: E402
    ATLAS_PATHS, assign_atlas_labels, assign_atlas_labels_with_hemisphere)
from homer.eval.nulls import translation_spin_null, _route_normalized         # noqa: E402

# Same seven markers, same layer assignment, as 01_layer_marker_validation.py
MARKERS = [("Cux1", "L2/3 upper"), ("Cux2", "L2/3 upper"), ("Satb2", "L2/3 upper"),
           ("Rorb", "L4 granular"),
           ("Fezf2", "L5/6 deep"), ("Tbr1", "L5/6 deep"), ("Foxp2", "L5/6 deep")]

N_SPIN = 1000
SEED = 0


def _z(v):
    v = np.asarray(v, float)
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / (s if s > 1e-9 else 1.0)


def _fill_nan_columnwise(E):
    E = E.copy()
    cm = np.nanmean(E, 0)
    idx = np.where(np.isnan(E))
    E[idx] = np.take(cm, idx[1])
    return E


def main():
    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = load_pi()
    prov = pi_provenance()
    print(f"π file: {prov['pi_file']}  sha256 {prov['pi_sha256'][:16]}…")

    mouse_coords = M.var[["x", "y", "z"]].to_numpy(float)
    me = _fill_nan_columnwise(np.load(ROOT / "data_external/mouse_genes.npy"))
    mg = pd.read_csv(ROOT / "data_external/mouse_gene_list.csv")
    he = _fill_nan_columnwise(np.load(ROOT / "data_external/human_genes.npy"))
    hg = pd.read_csv(ROOT / "data_external/human_gene_list.csv")

    # Cortical mask, defined exactly as 03_areal_type_reframe.py defines it
    sch = assign_atlas_labels(H.var, "schaefer_400", str(ROOT / ATLAS_PATHS["schaefer_400"]))
    sch = assign_atlas_labels_with_hemisphere(H.var, sch)
    cortex = sch > 0
    print(f"cortical human parcels: {int(cortex.sum())}/{len(H.var)}")

    rng = np.random.default_rng(42)
    rows = []
    for symbol, layer in MARKERS:
        mm = mg[mg["gene_symbol"].astype(str).str.lower().eq(symbol.lower())]
        hm = hg[hg["gene_symbol"].astype(str).str.upper().eq(symbol.upper())]
        if not len(mm) or not len(hm):
            print(f"  {symbol:<7s} missing in one species, skipped")
            continue
        m_vec = me[:, int(mm.iloc[0].name)]
        h_obs = he[:, int(hm.iloc[0].name)]

        # ---- as published: whole brain, permuted-pi null -----------------------------------
        pred_wb = _z(m_vec) @ pi
        h_z = _z(h_obs)
        r_wb = float(pearsonr(pred_wb, h_z)[0])
        null_wb = []
        for _ in range(200):
            null_wb.append(pearsonr(_z(m_vec) @ pi[rng.permutation(pi.shape[0])], h_z)[0])
        p_wb = float((np.asarray(null_wb) >= r_wb).mean())

        # ---- like for like: cortex only, translation spin null ------------------------------
        pred = _route_normalized(m_vec, pi)
        ok = cortex & np.isfinite(pred) & np.isfinite(h_obs)
        r_ctx = float(pearsonr(pred[ok], h_obs[ok])[0])
        spin = translation_spin_null(m_vec, np.where(cortex, h_obs, np.nan),
                                     pi, mouse_coords, n_trials=N_SPIN, seed=SEED)
        p_ctx = float(spin["p_translation_spin"])

        rows.append({"gene": symbol, "layer": layer,
                     "as_published": {"pearson_r": r_wb, "n": int(len(h_z)),
                                      "mask": "whole brain", "null": "permuted pi (200)",
                                      "p": p_wb},
                     "like_for_like": {"pearson_r": r_ctx, "n": int(ok.sum()),
                                       "mask": "Schaefer-400 cortex", "null": f"translation spin ({N_SPIN})",
                                       "p": p_ctx, "null_abs_mean": float(spin["null_abs_mean"])}})
        print(f"  {symbol:<7s} {layer:<12s} published r={r_wb:+.3f} p={p_wb:.3f}   "
              f"like-for-like r={r_ctx:+.3f} spin p={p_ctx:.3f}")

    pub = np.array([r["as_published"]["pearson_r"] for r in rows])
    lfl = np.array([r["like_for_like"]["pearson_r"] for r in rows])
    n_pub = int(sum(r["as_published"]["p"] < 0.05 for r in rows))
    n_lfl = int(sum(r["like_for_like"]["p"] < 0.05 for r in rows))

    out = {
        "_what": ("The seven Hodge individual layer markers scored two ways: as published "
                  "(whole brain, permuted-pi null) and like-for-like with the layer contrasts of "
                  "03_areal_type_reframe.py (Schaefer-400 cortex, translation spin null). Section 3 "
                  "compares the marker mean against the contrast mean; only the like-for-like "
                  "column is comparable to the contrasts."),
        "_compare_against": "outputs/logs/hodge_areal_type_reframe.json (contrasts, cortex + spin)",
        "n_markers": len(rows),
        "as_published": {"mean_pearson_r": float(pub.mean()), "n_significant": n_pub,
                         "matches_manuscript_0.23": bool(abs(pub.mean() - 0.228) < 0.02)},
        "like_for_like": {"mean_pearson_r": float(lfl.mean()), "n_significant": n_lfl},
        "markers": rows,
        **prov,
    }
    p = ROOT / "outputs/logs/hodge_markers_like_for_like.json"
    p.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 74)
    print(f"as published    mean r = {pub.mean():+.3f}   {n_pub}/{len(rows)} significant "
          f"(whole brain, permuted-pi)")
    print(f"like for like   mean r = {lfl.mean():+.3f}   {n_lfl}/{len(rows)} significant "
          f"(cortex, translation spin)")
    print(f"layer contrasts mean r = +0.067   1/4 significant   "
          f"(cortex, translation spin; hodge_areal_type_reframe.json + refined)")
    print("\nIf the like-for-like marker mean is close to +0.07, section 3's areal-versus-laminar")
    print("dissociation is carried by the mask and the null, not by the biology. Read it before")
    print("editing that paragraph.")
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
