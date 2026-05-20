"""HOMER × BICCN cell-type marker cross-species validation.

BICCN's atlases ([Yao et al. 2023, Nature](https://www.nature.com/articles/s41586-023-06812-z)
for mouse; [Siletti et al. 2023, Science](https://www.science.org/doi/10.1126/science.add7046)
for human) establish that cell types are broadly conserved across mouse and
human brain, with cell-type-defining markers (Pvalb for parvalbumin
interneurons, Sst for somatostatin interneurons, Camk2a for excitatory
neurons, Gfap for astrocytes, etc.) maintaining their spatial distributions
across species.

This tests whether HOMER's π preserves cell-type marker spatial patterns
across species. It's parallel to the Hodge 2019 layer-marker test
(experiments/hodge_2019_cortical_layers) but tests cell-type-defining markers
instead of cortical-layer markers.

Hypothesis: cell-type markers (especially interneuron class markers) should
translate cross-species *better* than layer markers, because:
  - Cell-type spatial distributions are largely AREA-SPECIFIC (Pvalb high in
    sensorimotor, Vip preferring associative cortex), so HOMER's area-level
    anchors should capture them.
  - Layer markers are WITHIN-AREA structure (laminar), which HOMER's anchors
    don't supervise.

Procedure (same as Hodge 2019 validation):
  1. For each cell-type marker, look up mouse and human per-parcel expression
     from HOMER's Allen ISH (mouse) and AHBA microarray (human) matrices.
  2. Translate mouse z-scored expression through π → predicted human pattern.
  3. Correlate predicted vs observed human expression (Pearson r over 2,094
     parcels, 200 permuted-π null trials).
  4. Aggregate by cell-type class (interneuron, glutamatergic, glia,
     neuromodulator) and report mean r per class.
  5. Side-by-side comparison vs Hodge 2019 layer-marker result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from homer.data import load_cached


# BICCN-aligned cell-type markers, grouped by class.
# All markers present in HOMER's 61-gene curated panel + AHBA microarray.
CELL_TYPE_MARKERS = {
    "Interneuron — Pvalb (parvalbumin)":   {"genes": ["Pvalb"],         "class": "interneuron"},
    "Interneuron — Sst (somatostatin)":    {"genes": ["Sst"],           "class": "interneuron"},
    "Interneuron — Vip (VIP+)":            {"genes": ["Vip"],           "class": "interneuron"},
    "Interneuron — Calb1 (calbindin)":     {"genes": ["Calb1"],         "class": "interneuron"},
    "Interneuron — Calb2 (calretinin)":    {"genes": ["Calb2"],         "class": "interneuron"},
    "Interneuron — Reln (reelin)":         {"genes": ["Reln"],          "class": "interneuron"},
    "Interneuron — Lhx6 (MGE TF)":         {"genes": ["Lhx6"],          "class": "interneuron"},
    "GABA synthesis — Gad1+Gad2":          {"genes": ["Gad1", "Gad2"],  "class": "gabaergic_synth"},
    "Glutamatergic — Camk2a":              {"genes": ["Camk2a"],        "class": "glutamatergic"},
    "Glutamatergic — Slc17a7 (Vglut1)":    {"genes": ["Slc17a7"],       "class": "glutamatergic"},
    "Glutamatergic — Slc17a6 (Vglut2)":    {"genes": ["Slc17a6"],       "class": "glutamatergic"},
    "NMDA — Grin1+Grin2a+Grin2b":          {"genes": ["Grin1", "Grin2a", "Grin2b"], "class": "glutamatergic"},
    "Astrocyte — Gfap":                    {"genes": ["Gfap"],          "class": "astrocyte"},
    "Astrocyte — Aqp4":                    {"genes": ["Aqp4"],          "class": "astrocyte"},
    "Oligodendrocyte — Mbp":               {"genes": ["Mbp"],           "class": "oligodendrocyte"},
    "Oligodendrocyte — Plp1":              {"genes": ["Plp1"],          "class": "oligodendrocyte"},
    "Oligodendrocyte TF — Olig2":          {"genes": ["Olig2"],         "class": "oligodendrocyte"},
    "Oligodendrocyte TF — Sox10":          {"genes": ["Sox10"],         "class": "oligodendrocyte"},
    "Microglia — Cx3cr1":                  {"genes": ["Cx3cr1"],        "class": "microglia"},
    "Dopaminergic — Th":                   {"genes": ["Th"],            "class": "dopaminergic"},
    "Dopaminergic — Slc6a3 (DAT)":         {"genes": ["Slc6a3"],        "class": "dopaminergic"},
    "Dopaminergic — Drd1":                 {"genes": ["Drd1"],          "class": "dopaminergic"},
    "Dopaminergic — Drd2":                 {"genes": ["Drd2"],          "class": "dopaminergic"},
    "Serotonergic — Tph2":                 {"genes": ["Tph2"],          "class": "serotonergic"},
    "Serotonergic — Slc6a4 (SERT)":        {"genes": ["Slc6a4"],        "class": "serotonergic"},
}


def _z(v):
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-9 else np.zeros_like(v)


def main():
    print("=" * 80)
    print("HOMER × BICCN cell-type marker cross-species validation")
    print("=" * 80)

    pi = np.load(ROOT / "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")
    print(f"π: {pi.shape}, mass {pi.sum():.4f}")

    # Mouse Allen ISH (1864 × 61 genes)
    mouse_expr = np.load(ROOT / "data_external/mouse_genes.npy")
    mouse_genes = pd.read_csv(ROOT / "data_external/mouse_gene_list.csv")
    col_mean = np.nanmean(mouse_expr, axis=0)
    inds = np.where(np.isnan(mouse_expr))
    mouse_expr = mouse_expr.copy()
    mouse_expr[inds] = np.take(col_mean, inds[1])

    # Human AHBA microarray (2094 × 15633 genes)
    human_expr = np.load(ROOT / "data_external/human_genes.npy")
    human_genes = pd.read_csv(ROOT / "data_external/human_gene_list.csv")
    cmean_h = np.nanmean(human_expr, axis=0)
    inds = np.where(np.isnan(human_expr))
    human_expr = human_expr.copy()
    human_expr[inds] = np.take(cmean_h, inds[1])

    print(f"Mouse ISH: {mouse_expr.shape}  Human AHBA: {human_expr.shape}")

    # Per-marker test
    print(f"\n{'Marker':<40s} | {'class':<18s} | {'r_pred_vs_obs':>14s} | "
          f"{'spearman':>10s} | {'null mean (CI)':>22s} | {'emp p':>6s}")
    print("-" * 130)

    rng = np.random.default_rng(seed=42)
    n_trials = 200
    results = []
    for marker_name, info in CELL_TYPE_MARKERS.items():
        genes = info["genes"]
        # Mouse: average z-scored expression of marker genes per parcel
        m_idxs = []
        for g in genes:
            mm = mouse_genes[mouse_genes["gene_symbol"].str.lower() == g.lower()]
            if len(mm) > 0:
                m_idxs.append(int(mm.iloc[0].name))
        if not m_idxs:
            print(f"  {marker_name:<40s} | (no mouse data — skip)")
            continue
        m_score = np.column_stack([_z(mouse_expr[:, i]) for i in m_idxs]).mean(axis=1)

        # Human: same
        h_idxs = []
        for g in genes:
            hh = human_genes[human_genes["gene_symbol"].str.upper() == g.upper()]
            if len(hh) > 0:
                h_idxs.append(int(hh.iloc[0].name))
        if not h_idxs:
            print(f"  {marker_name:<40s} | (no human data — skip)")
            continue
        h_score = np.column_stack([_z(human_expr[:, i]) for i in h_idxs]).mean(axis=1)
        h_obs = _z(h_score)

        # Translate through π
        pred_h = m_score @ pi
        r_p, p_p = pearsonr(pred_h, h_obs)
        r_s, _ = spearmanr(pred_h, h_obs)

        # Permuted-π null
        null_rs = []
        for _ in range(n_trials):
            perm = rng.permutation(pi.shape[0])
            pred_n = m_score @ pi[perm]
            r_n, _ = pearsonr(pred_n, h_obs)
            null_rs.append(r_n)
        null_rs = np.array(null_rs)
        emp_p = float((null_rs >= r_p).mean())

        ci_lo = np.percentile(null_rs, 2.5)
        ci_hi = np.percentile(null_rs, 97.5)
        print(f"  {marker_name:<40s} | {info['class']:<18s} | {r_p:>+14.3f} | "
              f"{r_s:>+10.3f} | {null_rs.mean():+.3f} ({ci_lo:+.3f}, {ci_hi:+.3f}) | "
              f"{emp_p:.3f}")

        results.append({
            "marker": marker_name, "class": info["class"], "genes": genes,
            "pearson_r": float(r_p), "pearson_p_analytical": float(p_p),
            "spearman_r": float(r_s),
            "null_mean": float(null_rs.mean()),
            "null_ci95": [float(ci_lo), float(ci_hi)],
            "empirical_p": emp_p,
        })

    # Aggregate by class
    print(f"\n{'='*80}")
    print(f"Aggregated by cell-type class:")
    print(f"-" * 80)
    classes = {}
    for r in results:
        classes.setdefault(r["class"], []).append(r)
    class_summary = []
    for cls in sorted(classes.keys()):
        rs = [r["pearson_r"] for r in classes[cls]]
        n_sig = sum(1 for r in classes[cls] if r["empirical_p"] < 0.05)
        n_total = len(classes[cls])
        mean_r = float(np.mean(rs))
        print(f"  {cls:<22s}: mean r = {mean_r:+.3f}  ({n_sig}/{n_total} markers emp p<0.05)")
        class_summary.append({
            "class": cls, "n_markers": int(n_total),
            "n_significant": int(n_sig), "mean_pearson_r": mean_r,
        })

    # Compare against Hodge 2019 layer-marker result
    hodge_path = ROOT / "outputs/logs/hodge_2019_layer_markers.json"
    if hodge_path.exists():
        hodge = json.loads(hodge_path.read_text())
        hodge_mean_r = float(np.mean([m["pearson_r"] for m in hodge["markers"]]))
        n_sig_hodge = sum(1 for m in hodge["markers"] if m["empirical_p"] < 0.05)
        n_hodge = len(hodge["markers"])
        print(f"\n{'='*80}")
        print(f"Side-by-side comparison:")
        print(f"-" * 80)
        biccn_mean_r = float(np.mean([r["pearson_r"] for r in results]))
        biccn_n_sig = sum(1 for r in results if r["empirical_p"] < 0.05)
        print(f"  BICCN cell-type markers (this test):     mean r = {biccn_mean_r:+.3f}  "
              f"({biccn_n_sig}/{len(results)} emp p<0.05)")
        print(f"  Hodge 2019 cortical layer markers:       mean r = {hodge_mean_r:+.3f}  "
              f"({n_sig_hodge}/{n_hodge} emp p<0.05)")
        if biccn_mean_r > hodge_mean_r + 0.05:
            verdict = (f"BICCN markers translate BETTER than layer markers "
                       f"(+{biccn_mean_r - hodge_mean_r:.3f} difference). "
                       f"Supports the hypothesis that area-specific cell-type "
                       f"distributions translate cross-species while within-area "
                       f"laminar markers don't.")
        elif biccn_mean_r > hodge_mean_r:
            verdict = (f"BICCN markers translate slightly better than layer "
                       f"markers (+{biccn_mean_r - hodge_mean_r:.3f}).")
        else:
            verdict = (f"BICCN markers do NOT translate better than layer markers "
                       f"({biccn_mean_r - hodge_mean_r:+.3f}). Suggests the cross-species "
                       f"resolution boundary is consistent across marker types.")
    else:
        verdict = "Hodge results not available for comparison."
    print(f"\nVerdict: {verdict}")

    out = {
        "n_markers": len(results),
        "per_marker": results,
        "per_class": class_summary,
        "verdict": verdict,
    }
    if hodge_path.exists():
        out["comparison"] = {
            "biccn_mean_pearson_r": biccn_mean_r,
            "biccn_n_significant": biccn_n_sig,
            "hodge_mean_pearson_r": hodge_mean_r,
            "hodge_n_significant": n_sig_hodge,
        }
    out_path = ROOT / "outputs" / "logs" / "biccn_2023_cell_types.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
