"""Disorder-UNIQUE gene-set test, a fair probe of cross-species specificity.

Phase 1 found that routing each disorder's FULL gene set through π gives
near-identical human predictions (off-diagonal r = +0.988), concluding "shared
psychiatric geometry, not disorder-specific". But the full sets overlap heavily
(autism 1,713 genes; SCZ 530; bipolar 109; ADHD 30, with large pairwise
intersections), so identical inputs trivially give identical outputs. That does
not actually test specificity.

Here we strip each disorder to the genes UNIQUE to it (present in that disorder's
set and NO other), route those through the same π, and recompute the
cross-disorder correlation matrix. If the unique predictions still correlate at
~0.99, the shared-geometry conclusion is robust. If they diverge, there
IS disorder-specific spatial information that the overlapping full sets washed out.

Usage:
    PYTHONPATH=src python experiments/enigma_cross_disorder/04_disorder_unique.py
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
sys.path.insert(0, str(ROOT / "experiments" / "enigma_cross_disorder"))

from otter.data import load_pi, pi_provenance  # noqa: E402

pd_mod = import_module("01_per_disorder_prediction")


def corr_matrix(preds, names):
    n = len(names)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = pearsonr(preds[names[i]], preds[names[j]])[0]
            M[i, j] = M[j, i] = r
    return M


def main():
    print("=" * 78)
    print("ENIGMA cross-disorder. DISORDER-UNIQUE gene-set specificity test")
    print("=" * 78)

    expr = np.load(ROOT / "experiments/autism_subtypes/allen_expansion/pagani_mouse_expr.npy")
    meta = pd.read_csv(ROOT / "experiments/autism_subtypes/allen_expansion/pagani_gene_list_resolved.csv")
    pi = load_pi()
    prov = pi_provenance()
    print(f"π: {pi.shape}  {prov['pi_file']}  sha256={prov['pi_sha256']}")
    expr = expr.copy()
    cmean = np.nanmean(expr, 0); nz = np.where(np.isnan(expr)); expr[nz] = np.take(cmean, nz[1])
    z = (expr - expr.mean(0, keepdims=True)) / (expr.std(0, keepdims=True) + 1e-9)
    gene_to_idx = {g.lower(): i for i, g in enumerate(meta["mouse_symbol"])}

    disorders = pd_mod.load_disorder_gene_sets()
    low = {d: {g.lower() for g in gs} for d, gs in disorders.items()}

    def route(gene_lower_set):
        idx = [gene_to_idx[g] for g in gene_lower_set if g in gene_to_idx]
        if len(idx) < 10:
            return None, len(idx)
        return z[:, idx].mean(1) @ pi, len(idx)

    # FULL-set predictions (reproduce Phase 1)
    full_pred, n_full = {}, {}
    print(f"\n{'disorder':<16}{'full overlap':>13}{'globally-unique overlap':>26}")
    for d in low:
        others = set().union(*[low[o] for o in low if o != d])
        fp, nf = route(low[d]); _, nu = route(low[d] - others)
        n_full[d] = nf
        if fp is not None: full_pred[d] = fp
        print(f"{d:<16}{nf:>13}{nu:>26}")

    names_full = [d for d in full_pred]
    Cf = corr_matrix(full_pred, names_full)
    iu_f = np.triu_indices(len(names_full), 1)
    print(f"\nFull-set cross-disorder off-diagonal r: mean {Cf[iu_f].mean():+.3f} "
          f"(min {Cf[iu_f].min():+.3f}, max {Cf[iu_f].max():+.3f})  [{len(names_full)} disorders]")
    print("(Note: the non-autism gene sets are essentially NESTED in the 1,713-gene autism set,\n"
          " so the full-set similarity partly reflects gene-set OVERLAP, not just smooth routing.)")

    # PAIRWISE relative-unique test: genes in A-not-B vs B-not-A, routed separately.
    # This is the fair specificity probe, it removes the shared genes for each pair.
    print(f"\nPairwise relative-unique test  (r of A-only vs B-only predicted maps):")
    pairwise = {}
    big = [d for d in names_full]
    for i in range(len(big)):
        for j in range(i + 1, len(big)):
            A, B = big[i], big[j]
            pa, na = route(low[A] - low[B])
            pb, nb = route(low[B] - low[A])
            if pa is None or pb is None:
                print(f"  {A[:10]:>10} vs {B[:10]:<10}  (A-only={na}, B-only={nb}; <10 → skip)")
                continue
            r = float(pearsonr(pa, pb)[0])
            pairwise[f"{A}|{B}"] = {"r": r, "n_A_only": na, "n_B_only": nb}
            tag = "DISTINCT" if r < 0.5 else ("partly" if r < 0.85 else "still ~identical")
            print(f"  {A[:10]:>10} vs {B[:10]:<10}  A-only={na:>4}  B-only={nb:>4}  r={r:+.3f}  → {tag}")

    out = {
        **prov,
        "n_full_overlap": n_full,
        "full_disorders": names_full,
        "full_offdiag_mean": float(Cf[iu_f].mean()),
        "full_corr_matrix": Cf.tolist(),
        "note": "non-autism gene sets are nested in the autism set; pairwise relative-unique is the fair probe",
        "pairwise_relative_unique": pairwise,
    }
    out_path = ROOT / "outputs" / "logs" / "enigma_disorder_unique.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
