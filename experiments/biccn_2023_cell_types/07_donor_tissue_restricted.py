#!/usr/bin/env python3
"""Donor-tissue-restricted scoring for the eleven gene-derived measures of Figure 3d.

The human side of these measures is Allen Human Brain Atlas expression pulled through abagen.
Only some human parcels contain donor tissue; the producers fill the remainder with each gene's
column mean before scoring, so the correlations they report run over their full territory even
though a large share of the target vector is an imputed constant. This script re-scores the same
measures over the parcels that carry donor tissue and writes both values side by side.

Eleven measures are covered, all drawing on the same two expression matrices:
  eight cell-class measures from 03_contrast_reframe.py and 05_composition_from_markers.py,
    scored over all 2,094 human parcels,
  three laminar contrasts from hodge_2019_cortical_layers/03_areal_type_reframe.py, scored over
    the Schaefer-400 cortical parcels.
All eleven are scored on the same footing.

The class definitions and the scoring functions are imported from the producers rather than
restated, so a change to any of them reaches this script. The published values are recomputed here
and checked against the committed logs before anything is written; a mismatch aborts, since it
would mean the measure definitions have moved and the restricted numbers would not be comparable
with the published ones.

The donor mask is taken from the raw expression matrix before imputation: a human parcel is kept
when every gene column is finite for it. The same test is applied on the mouse side and reported,
though mouse ISH coverage is near complete.

Significance is the translation spin null used by the producers, rotated on the mouse sphere and
routed through the real coupling, with the correlation taken over the restricted parcels only.

Run: cd otter && PYTHONPATH=src python experiments/biccn_2023_cell_types/07_donor_tissue_restricted.py
Writes outputs/logs/gene_maps_donor_tissue_restricted.json
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import hypergeom, pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from otter.data import load_cached, load_pi, pi_provenance          # noqa: E402
from otter.data.atlas_regions import (                              # noqa: E402
    ATLAS_PATHS, assign_atlas_labels, assign_atlas_labels_with_hemisphere)
from otter.eval.nulls import _route_normalized, _haar_rotation      # noqa: E402

c3 = import_module("03_contrast_reframe")
c5 = import_module("05_composition_from_markers")

_hodge_path = ROOT / "experiments/hodge_2019_cortical_layers/03_areal_type_reframe.py"
_spec = spec_from_file_location("hodge_areal", _hodge_path)
hz = module_from_spec(_spec); _spec.loader.exec_module(hz)

# Whole-brain values as published, from the two committed logs. Recomputed and checked below.
EXPECTED = {
    "glutamatergic": 0.302229582215497,
    "GABAergic": 0.00308187413472899,
    "astrocyte": 0.24672837920814938,
    "oligodendrocyte": 0.0699784607161994,
    "microglia": -0.032006131942705685,
    "excitatory_minus_inhibitory": 0.3401680729442248,
    "neuronal_minus_glial": 0.3538605350012064,
    "dopaminergic_hotspot": 0.17354981305072653,
    # the three laminar contrasts, from hodge_areal_type_reframe.json; these are already
    # cortex-restricted, so their base value is the cortical one, not a whole-brain one
    "supragranular_minus_infragranular": 0.008084976451264637,
    "granular_L4_minus_infragranular": 0.18712178577391028,
    "supragranular_minus_granular": 0.021039663620110366,
}
# measures whose published value is scored over Schaefer-400 cortex rather than all parcels
CORTEX_ONLY = ("supragranular_minus_infragranular", "granular_L4_minus_infragranular",
               "supragranular_minus_granular")
TOL = 1e-9
N_SPIN = 1000


def build_measures(m_expr, m_genes, h_expr, h_genes):
    """The eight measures, each built exactly as its own producer builds it."""

    def s3(c, human):
        expr, gdf = (h_expr, h_genes) if human else (m_expr, m_genes)
        return c3.class_score(expr, gdf, c3.CLASS_MARKERS[c], "gene_symbol", human)

    def s5(c, human):
        expr, gdf = (h_expr, h_genes) if human else (m_expr, m_genes)
        return c5.class_score(expr, gdf, c5.CLASS_MARKERS[c])

    meas = {}
    # five composition classes, from 05_composition_from_markers.py
    for c in c5.CLASS_MARKERS:
        meas[c] = (s5(c, False), s5(c, True))
    # three contrast/hotspot measures, from 03_contrast_reframe.py
    meas["excitatory_minus_inhibitory"] = (
        s3("glutamatergic", False) - s3("interneuron", False),
        s3("glutamatergic", True) - s3("interneuron", True))
    neu, gli = ["glutamatergic", "interneuron"], ["astrocyte", "oligodendrocyte", "microglia"]
    meas["neuronal_minus_glial"] = (
        np.mean([s3(c, False) for c in neu], 0) - np.mean([s3(c, False) for c in gli], 0),
        np.mean([s3(c, True) for c in neu], 0) - np.mean([s3(c, True) for c in gli], 0))
    meas["dopaminergic_hotspot"] = (s3("dopaminergic", False), s3("dopaminergic", True))

    # three laminar contrasts, from hodge_2019_cortical_layers/03_areal_type_reframe.py
    def sh(genes, human):
        expr, gdf = (h_expr, h_genes) if human else (m_expr, m_genes)
        return hz.score(expr, gdf, genes, human)

    for name, (pos, neg) in (
            ("supragranular_minus_infragranular", (hz.UPPER, hz.DEEP)),
            ("granular_L4_minus_infragranular", (hz.GRANULAR, hz.DEEP)),
            ("supragranular_minus_granular", (hz.UPPER, hz.GRANULAR))):
        meas[name] = (sh(pos, False) - sh(neg, False), sh(pos, True) - sh(neg, True))
    return meas


def spin_p(mvec, hvec, pi, sphere, mask, r_obs, n_spin, seed):
    """Translation spin null, correlation taken over `mask` only."""
    rng = np.random.default_rng(seed)
    null = np.empty(n_spin)
    for i in range(n_spin):
        _, perm = cKDTree(sphere @ _haar_rotation(rng).T).query(sphere)
        p = _route_normalized(mvec[perm], pi)
        m = np.isfinite(p) & np.isfinite(hvec) & mask
        null[i] = pearsonr(p[m], hvec[m])[0]
    return (float((np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_spin + 1)),
            float(np.mean(np.abs(null))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spin", type=int, default=N_SPIN,
                    help="translation-spin trials per measure (default %d)" % N_SPIN)
    args = ap.parse_args()

    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = load_pi()
    prov = pi_provenance()
    print("pi: %s  sha256 %s" % (prov["pi_file"], prov["pi_sha256"]))
    mouse_coords = M.var[["x", "y", "z"]].to_numpy(float)

    m_expr = np.load(ROOT / "data_external/mouse_genes.npy")
    m_genes = pd.read_csv(ROOT / "data_external/mouse_gene_list.csv")
    h_expr = np.load(ROOT / "data_external/human_genes.npy")
    h_genes = pd.read_csv(ROOT / "data_external/human_gene_list.csv")

    # Coverage masks, taken before imputation.
    has_donor = np.isfinite(h_expr).all(1)
    has_ish = np.isfinite(m_expr).all(1)
    print("human parcels with donor tissue: %d/%d" % (has_donor.sum(), len(has_donor)))
    print("mouse parcels with complete ISH: %d/%d" % (has_ish.sum(), len(has_ish)))
    if has_donor.sum() == len(has_donor):
        raise SystemExit(
            "every human parcel is finite, so the raw matrix has already been imputed. This "
            "script needs data_external/human_genes.npy as abagen produced it.")

    # Impute exactly as the producers do, so the published-value check is like for like.
    for E in (m_expr, h_expr):
        cm = np.nanmean(E, 0)
        idx = np.where(np.isnan(E))
        E[idx] = np.take(cm, idx[1])

    measures = build_measures(m_expr, m_genes, h_expr, h_genes)
    assert len(measures) == 11, len(measures)

    # Schaefer-400 cortex mask, built exactly as the laminar producer builds it
    sch = assign_atlas_labels(H.var, "schaefer_400", str(ROOT / ATLAS_PATHS["schaefer_400"]))
    sch = assign_atlas_labels_with_hemisphere(H.var, sch)
    cortex = sch > 0
    print("human parcels in Schaefer-400 cortex: %d/%d" % (cortex.sum(), len(cortex)))
    print("of which carry donor tissue: %d" % int((cortex & has_donor).sum()))

    c = mouse_coords - np.nanmean(mouse_coords, 0)
    sphere = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)

    out = dict(prov)
    out["_what"] = (
        "Donor-tissue-restricted scoring for the eleven gene-derived measures of Figure 3d. "
        "published_r is scored over each measure's own territory, all 2,094 parcels for the eight "
        "cell-class measures and the Schaefer-400 cortical parcels for the three laminar "
        "contrasts, and is recomputed here and checked against biccn_composition_from_markers.json, "
        "biccn_contrast_reframe.json and hodge_areal_type_reframe.json before anything is written. "
        "Only the donor-tissue parcels carry measured AHBA expression; in the published values the "
        "rest are filled with each gene's column mean, which pulls the correlation toward zero. "
        "donor_only_r scores the same routed mouse map over the donor-tissue parcels of that "
        "territory alone, against a translation spin null restricted the same way.")
    out["_masks"] = {
        "n_human_with_donor_tissue": int(has_donor.sum()),
        "n_human_total": int(len(has_donor)),
        "n_human_schaefer_cortex": None,          # filled below
        "n_human_cortex_with_donor_tissue": None, # filled below
        "n_mouse_with_complete_ish": int(has_ish.sum()),
        "n_mouse_total": int(len(has_ish)),
        "human_source": "data_external/human_genes.npy (Allen Human Brain Atlas via abagen)",
        "mouse_source": "data_external/mouse_genes.npy (Allen ISH energy at 200 um)",
    }
    out["_n_spin"] = int(args.n_spin)

    out["_masks"]["n_human_schaefer_cortex"] = int(cortex.sum())
    out["_masks"]["n_human_cortex_with_donor_tissue"] = int((cortex & has_donor).sum())

    for j, (name, (mv, hv)) in enumerate(measures.items()):
        # each measure is compared on its own published territory: all parcels for the
        # cell-class measures, Schaefer-400 cortex for the laminar contrasts
        base = cortex if name in CORTEX_ONLY else np.ones(len(hv), bool)
        pred = _route_normalized(mv, pi)
        ok = base & np.isfinite(pred) & np.isfinite(hv)
        published = float(pearsonr(pred[ok], hv[ok])[0])
        exp = EXPECTED[name]
        if abs(published - exp) > TOL:
            raise SystemExit(
                "%s: recomputed r %.12f does not match the committed %.12f. The measure "
                "definitions have moved, so the restricted value would not be comparable with the "
                "published one. Re-run the producer and update EXPECTED before using this script."
                % (name, published, exp))
        okd = ok & has_donor
        donor = float(pearsonr(pred[okd], hv[okd])[0])
        p, nullmean = spin_p(mv, hv, pi, sphere, ok & has_donor, donor, args.n_spin, seed=j)
        rec = {"territory": "schaefer_cortex" if name in CORTEX_ONLY else "all_parcels",
               "published_r": published, "donor_only_r": donor, "donor_only_spin_p": p,
               "donor_only_spin_null_abs_mean": nullmean,
               "n_parcels_published": int(ok.sum()), "n_parcels_donor": int(okd.sum())}
        if name == "dopaminergic_hotspot":
            # the hotspot is also reported as a top-decile overlap; recompute it restricted
            idx = np.where(okd)[0]
            n = len(idx)
            k = max(1, n // 10)
            top_pred = set(idx[np.argsort(pred[idx])[-k:]])
            top_obs = set(idx[np.argsort(hv[idx])[-k:]])
            ov = len(top_pred & top_obs)
            rec["donor_only_top_decile"] = {
                "n": int(n), "k": int(k), "overlap": int(ov),
                "hypergeometric_p": float(hypergeom.sf(ov - 1, n, k, k))}
        out[name] = rec
        print("  %-36s published %+.3f (n=%4d)   donor-only %+.3f (n=%4d, spin p = %.3f)"
              % (name, published, ok.sum(), donor, okd.sum(), p))

    dest = ROOT / "outputs/logs/gene_maps_donor_tissue_restricted.json"
    dest.write_text(json.dumps(out, indent=2))
    print("\nwrote %s" % dest.relative_to(ROOT))


if __name__ == "__main__":
    main()
