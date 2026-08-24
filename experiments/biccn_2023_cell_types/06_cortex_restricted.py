#!/usr/bin/env python3
"""Cortex-restricted sensitivity for the cell-class marker measures.

The eight cell-class marker maps of 03_contrast_reframe.py and
05_composition_from_markers.py are scored over all 2,094 human parcels, so part of the
correspondence they report is the contrast between cortex and subcortex rather than gradation
within the cortex. This script re-scores four of them over cortical parcels only, and over the cortical parcels that
carry AHBA donor tissue, and writes the values side by side.

The class definitions and the scoring functions are imported from the two sibling scripts rather
than restated, so a change to either reaches this script. The whole-brain values are recomputed
here and checked against the committed logs before anything is written; a mismatch aborts, since
it would mean the class definitions have moved and the cortex-only numbers would not be
comparable with the published whole-brain ones.

Cortex masks:
  mouse   isocortex parcels, from the Allen structure name in M.var["region_vote_ns_aba"]
  human   parcels whose MNI coordinate falls within RADIUS_MM of a labelled voxel of the
          Harvard-Oxford hemisphere-split cortical atlas

Run: cd otter && PYTHONPATH=src python experiments/biccn_2023_cell_types/06_cortex_restricted.py
Writes outputs/logs/biccn_cortex_restricted.json
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from otter.data import load_cached, load_pi, pi_provenance          # noqa: E402
from otter.eval.nulls import _route_normalized                      # noqa: E402

c3 = import_module("03_contrast_reframe")
c5 = import_module("05_composition_from_markers")

RADIUS_MM = 6.0
HO_CORTL = ROOT / "data_external/_domhof_extracted/HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz"

# Whole-brain values as published, from the two committed logs. Recomputed and checked below.
EXPECTED = {
    "glutamatergic": 0.302229582215497,
    "excitatory_minus_inhibitory": 0.3401680729442248,
    "neuronal_minus_glial": 0.3538605350012064,
    "dopaminergic_hotspot": 0.17354981305072653,
}
TOL = 1e-9

# Allen structure-name stems that denote isocortex. The hippocampal formation, olfactory areas,
# cortical subplate, thalamus, hypothalamus, striatum, pallidum, midbrain, pons, medulla and
# cerebellum are all excluded.
ISOCORTEX_STEMS = (
    "primary motor area", "secondary motor area", "primary somatosensory area",
    "supplemental somatosensory area", "primary visual area", "visual area",
    "anterolateral visual area", "anteromedial visual area", "lateral visual area",
    "posterolateral visual area", "posteromedial visual area", "rostrolateral visual area",
    "primary auditory area", "dorsal auditory area", "ventral auditory area",
    "posterior auditory area", "anterior cingulate area", "prelimbic area", "infralimbic area",
    "orbital area", "agranular insular area", "gustatory area", "visceral area",
    "retrosplenial area", "posterior parietal association areas", "temporal association areas",
    "perirhinal area", "ectorhinal area", "frontal pole", "dorsal peduncular area",
)


def mouse_isocortex_mask(M) -> np.ndarray:
    names = M.var["region_vote_ns_aba"].astype(str).str.lower().fillna("")
    return names.apply(lambda s: any(s.startswith(k) for k in ISOCORTEX_STEMS)).to_numpy()


def human_cortex_mask(H) -> np.ndarray:
    import nibabel as nib
    from scipy.spatial import cKDTree

    if not HO_CORTL.exists():
        raise SystemExit(
            "missing %s\nFetch it with nilearn.datasets.fetch_atlas_harvard_oxford"
            "('cortl-maxprob-thr25-2mm') and place it there." % HO_CORTL)
    im = nib.load(str(HO_CORTL))
    dat = np.asarray(im.dataobj)
    ijk = np.argwhere(dat > 0)
    xyz = nib.affines.apply_affine(im.affine, ijk)
    d, _ = cKDTree(xyz).query(H.var[["x", "y", "z"]].to_numpy(float))
    return d <= RADIUS_MM


def main():
    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = load_pi()
    prov = pi_provenance()
    print("pi: %s  sha256 %s" % (prov["pi_file"], prov["pi_sha256"]))

    m_expr = np.load(ROOT / "data_external/mouse_genes.npy")
    m_genes = pd.read_csv(ROOT / "data_external/mouse_gene_list.csv")
    h_expr = np.load(ROOT / "data_external/human_genes.npy")
    h_genes = pd.read_csv(ROOT / "data_external/human_gene_list.csv")
    # donor-tissue mask, taken before imputation
    has_donor = np.isfinite(h_expr).all(1)
    for E in (m_expr, h_expr):
        cm = np.nanmean(E, 0); idx = np.where(np.isnan(E)); E[idx] = np.take(cm, idx[1])

    def s3(c, human):
        expr, gdf = (h_expr, h_genes) if human else (m_expr, m_genes)
        return c3.class_score(expr, gdf, c3.CLASS_MARKERS[c], "gene_symbol", human)

    def s5(c, human):
        expr, gdf = (h_expr, h_genes) if human else (m_expr, m_genes)
        return c5.class_score(expr, gdf, c5.CLASS_MARKERS[c])

    # The four measures, each built exactly as its own producer builds it.
    measures = {}
    measures["glutamatergic"] = (s5("glutamatergic", False), s5("glutamatergic", True))
    measures["excitatory_minus_inhibitory"] = (
        s3("glutamatergic", False) - s3("interneuron", False),
        s3("glutamatergic", True) - s3("interneuron", True))
    neu, gli = ["glutamatergic", "interneuron"], ["astrocyte", "oligodendrocyte", "microglia"]
    measures["neuronal_minus_glial"] = (
        np.mean([s3(c, False) for c in neu], 0) - np.mean([s3(c, False) for c in gli], 0),
        np.mean([s3(c, True) for c in neu], 0) - np.mean([s3(c, True) for c in gli], 0))
    measures["dopaminergic_hotspot"] = (s3("dopaminergic", False), s3("dopaminergic", True))

    hum_cx = human_cortex_mask(H)
    mou_cx = mouse_isocortex_mask(M)
    print("cortical parcels: mouse %d/%d, human %d/%d (radius %.0f mm)"
          % (mou_cx.sum(), len(mou_cx), hum_cx.sum(), len(hum_cx), RADIUS_MM))
    print("human parcels with donor tissue: %d/%d, of which cortical: %d"
          % (has_donor.sum(), len(has_donor), int((hum_cx & has_donor).sum())))

    out = dict(prov)
    out["_what"] = (
        "Cortex-restricted sensitivity for four cell-class marker measures. whole_brain_r is "
        "recomputed here and checked against biccn_composition_from_markers.json and "
        "biccn_contrast_reframe.json. cortex_only_r scores the same routed mouse map over "
        "cortical human parcels alone. cortex_only_both_sides additionally drops non-isocortical "
        "mouse parcels and renormalises the coupling over the survivors.")
    out["_masks"] = {"human_radius_mm": RADIUS_MM, "n_human_cortical": int(hum_cx.sum()),
                     "n_mouse_isocortex": int(mou_cx.sum()),
                     "n_human_with_donor_tissue": int(has_donor.sum()),
                     "n_human_cortical_with_donor_tissue": int((hum_cx & has_donor).sum()),
                     "human_atlas": str(HO_CORTL.relative_to(ROOT))}

    pi_cx = pi[mou_cx]
    for name, (mv, hv) in measures.items():
        pred = _route_normalized(mv, pi)
        ok = np.isfinite(pred) & np.isfinite(hv)
        whole = float(pearsonr(pred[ok], hv[ok])[0])
        okc = ok & hum_cx
        cortex = float(pearsonr(pred[okc], hv[okc])[0])
        pred_b = _route_normalized(mv[mou_cx], pi_cx)
        okb = np.isfinite(pred_b) & np.isfinite(hv) & hum_cx
        both = float(pearsonr(pred_b[okb], hv[okb])[0])
        exp = EXPECTED[name]
        if abs(whole - exp) > TOL:
            raise SystemExit(
                "%s: recomputed whole-brain r %.12f does not match the committed %.12f. The class "
                "definitions have moved, so the cortex-only value would not be comparable with the "
                "published one. Re-run the producer and update EXPECTED before using this script."
                % (name, whole, exp))
        okcd = okc & has_donor
        cortex_donor = float(pearsonr(pred[okcd], hv[okcd])[0])
        out[name] = {"whole_brain_r": whole, "cortex_only_r": cortex,
                     "cortex_only_both_sides_r": both,
                     "cortex_and_donor_r": cortex_donor,
                     "n_parcels_whole": int(ok.sum()), "n_parcels_cortex": int(okc.sum()),
                     "n_parcels_cortex_donor": int(okcd.sum())}
        print("  %-30s whole %+.3f  cortex %+.3f  both-sides %+.3f  cortex+donor %+.3f (n=%d)"
              % (name, whole, cortex, both, cortex_donor, okcd.sum()))

    dest = ROOT / "outputs/logs/biccn_cortex_restricted.json"
    dest.write_text(json.dumps(out, indent=2))
    print("\nwrote %s" % dest.relative_to(ROOT))


if __name__ == "__main__":
    main()
