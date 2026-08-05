"""Contrast / hotspot reframe of the BICCN cell-type-marker test.

The per-marker test correlates each gene's smooth z-scored expression map
mouse→human (mean r=+0.089, weak). Two problems make that the wrong question:
single-gene maps are noisy, and any two smooth cortical maps share spatial
autocorrelation. Here we ask sharper, OTTER-strong-mode questions:

  1. CLASS CONTRASTS (magnitude-cancelling, like the Pagani contrast). Build a
     per-parcel class score (mean z over a class's markers) for both species and
     test the *contrast* between classes, excitatory−inhibitory (Glut − Intern)
     and neuronal−glial, which removes the shared "everything is high in cortex"
     baseline. Scored against the FAIR translation-spin null (spin the mouse map
     on the mouse sphere, route through the real π).
  2. SUBCORTICAL HOTSPOT (discrete). Dopaminergic markers (Th/Drd1/Drd2/Slc6a3)
     peak in striatum/midbrain. Does the mouse dopaminergic hotspot route through
     π to the human dopaminergic hotspot? Measured as top-decile overlap
     (hypergeometric) and hotspot-centroid agreement.

Usage:
    PYTHONPATH=src python experiments/biccn_2023_cell_types/03_contrast_reframe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, hypergeom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance      # noqa: E402
from otter.eval.nulls import translation_spin_null, _route_normalized  # noqa: E402

CLASS_MARKERS = {
    "glutamatergic": ["Camk2a", "Slc17a7", "Slc17a6", "Grin1", "Grin2a", "Grin2b"],
    "interneuron":   ["Pvalb", "Sst", "Vip", "Calb1", "Calb2", "Reln", "Lhx6",
                      "Gad1", "Gad2"],
    "astrocyte":     ["Gfap", "Aqp4"],
    "oligodendrocyte": ["Mbp", "Plp1", "Olig2", "Sox10"],
    "microglia":     ["Cx3cr1"],
    "dopaminergic":  ["Th", "Drd1", "Drd2", "Slc6a3"],
}


def _z(v):
    v = v.astype(float)
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / (s if s > 1e-9 else 1.0)


def class_score(expr, gene_df, genes, sym_col, upper):
    cols = []
    for g in genes:
        key = g.upper() if upper else g.lower()
        match = gene_df[gene_df[sym_col].astype(str).str.upper().eq(g.upper())]
        if len(match):
            cols.append(_z(expr[:, int(match.iloc[0].name)]))
    return np.column_stack(cols).mean(1) if cols else None


def main():
    print("=" * 78)
    print("BICCN cell types. CONTRAST + HOTSPOT reframe vs a fair spin null")
    print("=" * 78)

    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    pi = load_pi()                      # canonical coupling (pi_canonical.npy)
    prov = pi_provenance()
    print(f"π file: {prov['pi_file']}  sha256 {prov['pi_sha256']}")
    mouse_coords = M.var[["x", "y", "z"]].to_numpy(float)

    mouse_expr = np.load(ROOT / "data_external/mouse_genes.npy")
    mouse_genes = pd.read_csv(ROOT / "data_external/mouse_gene_list.csv")
    human_expr = np.load(ROOT / "data_external/human_genes.npy")
    human_genes = pd.read_csv(ROOT / "data_external/human_gene_list.csv")
    for E in (mouse_expr, human_expr):
        cm = np.nanmean(E, 0); idx = np.where(np.isnan(E)); E[idx] = np.take(cm, idx[1])

    def m_class(c):
        return class_score(mouse_expr, mouse_genes, CLASS_MARKERS[c], "gene_symbol", False)

    def h_class(c):
        return class_score(human_expr, human_genes, CLASS_MARKERS[c], "gene_symbol", True)

    classes = {c: (m_class(c), h_class(c)) for c in CLASS_MARKERS}

    # ---------- Test 1: class contrasts ----------
    contrasts = {
        "excitatory_minus_inhibitory": ("glutamatergic", "interneuron"),
        "neuronal_minus_glial": (["glutamatergic", "interneuron"],
                                  ["astrocyte", "oligodendrocyte", "microglia"]),
    }
    results = dict(prov)
    print(f"\n[1] Class contrasts (routed mouse contrast vs human contrast):")
    for name, (pos, neg) in contrasts.items():
        pos = [pos] if isinstance(pos, str) else pos
        neg = [neg] if isinstance(neg, str) else neg
        m_vec = np.mean([classes[c][0] for c in pos], 0) - np.mean([classes[c][0] for c in neg], 0)
        h_vec = np.mean([classes[c][1] for c in pos], 0) - np.mean([classes[c][1] for c in neg], 0)
        pred = _route_normalized(m_vec, pi)
        ok = np.isfinite(pred) & np.isfinite(h_vec)
        r = float(pearsonr(pred[ok], h_vec[ok])[0])
        spin = translation_spin_null(m_vec, h_vec, pi, mouse_coords, n_trials=1000, seed=0)
        sp = spin["p_translation_spin"]
        verdict = "SURVIVES" if sp < 0.05 else "n.s."
        print(f"    {name:<32} r={r:+.3f}  spin p={sp:.3f}  → {verdict}")
        results[name] = {"pearson_r": r, "spin_p": sp,
                         "spin_null_abs_mean": spin["null_abs_mean"]}

    # ---------- Test 2: subcortical dopaminergic hotspot ----------
    print(f"\n[2] Dopaminergic hotspot (top-decile overlap after routing):")
    m_dop, h_dop = classes["dopaminergic"]
    pred_dop = _route_normalized(m_dop, pi)
    ok = np.isfinite(pred_dop) & np.isfinite(h_dop)
    n = int(ok.sum())
    k = max(1, n // 10)                                   # top decile
    pred_top = set(np.argsort(np.where(ok, pred_dop, -np.inf))[-k:])
    obs_top = set(np.argsort(np.where(ok, h_dop, -np.inf))[-k:])
    overlap = len(pred_top & obs_top)
    # hypergeometric: P(overlap >= observed) drawing k from n with k successes
    p_hyper = float(hypergeom.sf(overlap - 1, n, k, k))
    r_dop = float(pearsonr(pred_dop[ok], h_dop[ok])[0])
    spin_dop = translation_spin_null(m_dop, h_dop, pi, mouse_coords, n_trials=1000, seed=1)
    sp_dop = spin_dop["p_translation_spin"]
    print(f"    top-decile overlap = {overlap}/{k}  (hypergeometric p = {p_hyper:.2e})")
    print(f"    full-map r = {r_dop:+.3f}  spin p = {sp_dop:.3f}")
    results["dopaminergic_hotspot"] = {
        "n_parcels": n, "top_decile_k": k, "overlap": overlap,
        "hypergeometric_p": p_hyper, "pearson_r": r_dop, "spin_p": sp_dop,
    }

    out_path = ROOT / "outputs" / "logs" / "biccn_contrast_reframe.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
