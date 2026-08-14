"""Cell-CLASS composition test through π, using the pre-aligned per-parcel gene
matrices (no atlas download / no coordinate mapping needed).

03 tested single class contrasts (E-I, neuronal-glial). This asks the compositional
question. A per-parcel profile over several cell classes is built for both species
(mean z over each class's markers), the mouse profile is routed through π, and two
quantities are tested:
  (1) per-class translation-spin, whether each class's map translates beyond spatial
      autocorrelation,
  (2) dominant-class agreement, whether π maps each human parcel to the cell class
      that dominates there, scored against a translation-spin null (rotate the mouse
      profiles on the mouse sphere, route through the real π, recompute).

Run: cd otter && PYTHONPATH=src python experiments/biccn_2023_cell_types/05_composition_from_markers.py
Writes outputs/logs/biccn_composition_from_markers.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance
from otter.eval.nulls import _route_normalized, _haar_rotation
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

CLASS_MARKERS = {
    "glutamatergic":  ["Camk2a", "Slc17a7", "Slc17a6", "Grin1", "Grin2a", "Grin2b"],
    "GABAergic":      ["Pvalb", "Sst", "Vip", "Calb1", "Calb2", "Reln", "Lhx6", "Gad1", "Gad2"],
    "astrocyte":      ["Gfap", "Aqp4"],
    "oligodendrocyte": ["Mbp", "Plp1", "Olig2", "Sox10"],
    "microglia":      ["Cx3cr1", "Ctss", "Csf1r"],
}
N_SPIN = 1000


def _z(v):
    v = v.astype(float); s = np.nanstd(v)
    return (v - np.nanmean(v)) / (s if s > 1e-9 else 1.0)


def class_score(expr, gene_df, genes):
    cols = []
    for g in genes:
        match = gene_df[gene_df["gene_symbol"].astype(str).str.upper().eq(g.upper())]
        if len(match):
            cols.append(_z(expr[:, int(match.iloc[0].name)]))
    return np.column_stack(cols).mean(1) if cols else None


def main():
    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    pi = load_pi()                      # canonical coupling (pi_canonical.npy)
    prov = pi_provenance()
    print(f"π file: {prov['pi_file']}  sha256 {prov['pi_sha256']}")
    mouse_coords = M.var[["x", "y", "z"]].to_numpy(float)

    m_expr = np.load(ROOT / "data_external/mouse_genes.npy")
    h_expr = np.load(ROOT / "data_external/human_genes.npy")
    m_genes = pd.read_csv(ROOT / "data_external/mouse_gene_list.csv")
    h_genes = pd.read_csv(ROOT / "data_external/human_gene_list.csv")
    for E in (m_expr, h_expr):
        cm = np.nanmean(E, 0); idx = np.where(np.isnan(E)); E[idx] = np.take(cm, idx[1])

    classes = list(CLASS_MARKERS)
    Mcomp = np.column_stack([class_score(m_expr, m_genes, CLASS_MARKERS[c]) for c in classes])   # (n_mouse, K)
    Hcomp = np.column_stack([class_score(h_expr, h_genes, CLASS_MARKERS[c]) for c in classes])   # (n_human, K)

    # predicted human composition = route each mouse class map through π
    Pred = np.column_stack([_route_normalized(Mcomp[:, j], pi) for j in range(len(classes))])
    ok = np.isfinite(Pred).all(1) & np.isfinite(Hcomp).all(1)

    # (1) per-class translation-spin
    def route_spin(mvec, hvec, n=N_SPIN, seed=0):
        c = mouse_coords - np.nanmean(mouse_coords, 0)
        sph = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
        r = pearsonr(*[a[np.isfinite(_route_normalized(mvec, pi)) & np.isfinite(hvec)]
                       for a in (_route_normalized(mvec, pi), hvec)])[0]
        rng = np.random.default_rng(seed); null = np.empty(n)
        for i in range(n):
            _, perm = cKDTree(sph @ _haar_rotation(rng).T).query(sph)
            p = _route_normalized(mvec[perm], pi); m = np.isfinite(p) & np.isfinite(hvec)
            null[i] = pearsonr(p[m], hvec[m])[0]
        return float(r), float((np.sum(np.abs(null) >= abs(r)) + 1) / (n + 1))

    print("[1] per-class translation spin:")
    per_class = {}
    for j, c in enumerate(classes):
        r, p = route_spin(Mcomp[:, j], Hcomp[:, j], seed=j)
        per_class[c] = {"r": r, "spin_p": p}
        print(f"    {c:<16} r={r:+.3f}  spin p={p:.3f}")

    # (2) dominant-class agreement + spin null
    obs_dom = Hcomp[ok].argmax(1)
    pred_dom = Pred[ok].argmax(1)
    agree = float((pred_dom == obs_dom).mean())
    c = mouse_coords - np.nanmean(mouse_coords, 0)
    sph = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(42); null = np.empty(N_SPIN)
    for i in range(N_SPIN):
        _, perm = cKDTree(sph @ _haar_rotation(rng).T).query(sph)
        Pr = np.column_stack([_route_normalized(Mcomp[perm, j], pi) for j in range(len(classes))])
        null[i] = (Pr[ok].argmax(1) == obs_dom).mean()
    p_agree = float((np.sum(null >= agree) + 1) / (N_SPIN + 1))
    chance = float(pd.Series(obs_dom).value_counts(normalize=True).max())   # majority-class baseline
    print(f"\n[2] dominant-class agreement = {agree:.3f}  (spin p = {p_agree:.4f}; majority baseline {chance:.3f})")

    out = {**prov, "classes": classes, "n_parcels_scored": int(ok.sum()),
           "per_class_translation_spin": per_class,
           "dominant_class_agreement": agree, "dominant_class_spin_p": p_agree,
           "majority_baseline": chance, "n_spin": N_SPIN}
    (ROOT / "outputs/logs/biccn_composition_from_markers.json").write_text(json.dumps(out, indent=2))
    print("\nwrote outputs/logs/biccn_composition_from_markers.json")


if __name__ == "__main__":
    main()
