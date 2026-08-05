"""Areal-type reframe of the Hodge cortical-layer-marker test.

Schaefer-400 cannot separate layers within an area, so the per-marker test
(mean r=+0.119) can only ever measure the *area-level* distribution of layer
genes, not lamination, the README says as much. Rather than present it as a
(impossible) laminar test, we recast it as the question it CAN answer and that
OTTER is good at: does π preserve cortical AREAL TYPE, the supragranular↔
infragranular (eulaminate↔agranular) axis that distinguishes sensory/granular
from limbic/agranular cortex? This is the same cytoarchitectural hierarchy that
Fulcher's T1w:T2w + Goulas type test captured and that DID survive a spin null.

We build the supragranular−infragranular contrast (upper L2/3 markers minus deep
L5/6 markers), high in granular sensory cortex, low in agranular cortex, and
the granular-L4 axis, route the mouse contrast through π, and test against the
FAIR translation-spin null, cortex-only.

Usage:
    PYTHONPATH=src python experiments/hodge_2019_cortical_layers/03_areal_type_reframe.py
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
from otter.data import load_cached, load_pi, pi_provenance       # noqa: E402
from otter.data.atlas_regions import (                           # noqa: E402
    ATLAS_PATHS, assign_atlas_labels, assign_atlas_labels_with_hemisphere)
from otter.eval.nulls import translation_spin_null, _route_normalized  # noqa: E402

UPPER = ["Cux1", "Cux2", "Satb2"]      # supragranular L2/3
GRANULAR = ["Rorb"]                     # L4
DEEP = ["Fezf2", "Tbr1", "Foxp2"]      # infragranular L5/6


def _z(v):
    v = v.astype(float); s = np.nanstd(v)
    return (v - np.nanmean(v)) / (s if s > 1e-9 else 1.0)


def score(expr, genes_df, genes, upper):
    cols = []
    for g in genes:
        match = genes_df[genes_df["gene_symbol"].astype(str).str.upper().eq(g.upper())]
        if len(match):
            cols.append(_z(expr[:, int(match.iloc[0].name)]))
    return np.column_stack(cols).mean(1) if cols else None


def main():
    print("=" * 78)
    print("Hodge layers. AREAL-TYPE reframe (supragranular↔infragranular) vs spin")
    print("=" * 78)

    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = load_pi()                      # canonical coupling (pi_canonical.npy)
    prov = pi_provenance()
    print(f"π file: {prov['pi_file']}  sha256 {prov['pi_sha256']}")
    mouse_coords = M.var[["x", "y", "z"]].to_numpy(float)

    me = np.load(ROOT / "data_external/mouse_genes.npy")
    mg = pd.read_csv(ROOT / "data_external/mouse_gene_list.csv")
    he = np.load(ROOT / "data_external/human_genes.npy")
    hg = pd.read_csv(ROOT / "data_external/human_gene_list.csv")
    for E in (me, he):
        cm = np.nanmean(E, 0); idx = np.where(np.isnan(E)); E[idx] = np.take(cm, idx[1])

    # cortical mask (human side) via Schaefer-400
    sch = assign_atlas_labels(H.var, "schaefer_400", str(ROOT / ATLAS_PATHS["schaefer_400"]))
    sch = assign_atlas_labels_with_hemisphere(H.var, sch)
    cortex = sch > 0
    print(f"cortical human parcels: {int(cortex.sum())}/{len(H.var)}")

    def m_score(genes): return score(me, mg, genes, False)
    def h_score(genes): return score(he, hg, genes, True)

    tests = {
        "supragranular_minus_infragranular": (UPPER, DEEP),
        "granular_L4_minus_infragranular":   (GRANULAR, DEEP),
        "supragranular_minus_granular":      (UPPER, GRANULAR),
    }
    results = dict(prov)
    for name, (pos, neg) in tests.items():
        m_vec = m_score(pos) - m_score(neg)
        h_vec = h_score(pos) - h_score(neg)
        pred = _route_normalized(m_vec, pi)
        ok = cortex & np.isfinite(pred) & np.isfinite(h_vec)
        r = float(pearsonr(pred[ok], h_vec[ok])[0])
        # fair spin null, cortex-restricted comparison
        spin = translation_spin_null(m_vec, np.where(cortex, h_vec, np.nan),
                                     pi, mouse_coords, n_trials=1000, seed=0)
        sp = spin["p_translation_spin"]
        verdict = "SURVIVES" if sp < 0.05 else "n.s."
        print(f"  {name:<36} r={r:+.3f}  spin p={sp:.3f}  → {verdict}")
        results[name] = {"pearson_r": r, "spin_p": sp,
                         "spin_null_abs_mean": spin["null_abs_mean"],
                         "n_cortical": int(ok.sum())}

    out_path = ROOT / "outputs" / "logs" / "hodge_areal_type_reframe.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
