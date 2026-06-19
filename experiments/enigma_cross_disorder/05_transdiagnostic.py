"""Transdiagnostic test — turn the 'no disorder-specificity' negative into a positive.

Phase 1 + the disorder-unique test (04) showed HOMER's per-disorder predictions are
near-identical (even disjoint gene sets give r≈0.98): HOMER carries a single SHARED
psychiatric spatial geometry, not disorder-specific biology. That is a real claim if
HOMER's generic map matches the actual transdiagnostic cortical signature — the
"p-factor" pattern of shared cortical vulnerability across disorders.

We test exactly that against ENIGMA observed cortical-thickness Cohen's d maps:
  • Build the ENIGMA TRANSDIAGNOSTIC AVERAGE = mean case-control Cohen's d across the
    four disorders we have gene sets for (ASD, SCZ, BD, ADHD), per DK region.
  • HOMER's GENERIC prediction = mean of its per-disorder routed maps (they're
    near-identical anyway), aggregated to the same DK regions.
  • Correlate HOMER-generic vs the ENIGMA transdiagnostic average, vs each disorder,
    and vs two held-out disorders not in HOMER's gene sets (MDD, OCD) — and test
    significance with a spin null over the 34 DK region centroids.

ENIGMA CSVs expected in data_external/enigma/cortical_thickness_<disorder>.csv
(from the ENIGMA Toolbox repo summary_statistics; Structure col + d_icv col).

Usage:
    PYTHONPATH=src python experiments/enigma_cross_disorder/05_transdiagnostic.py
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "enigma_cross_disorder"))

from homer.data import load_cached                     # noqa: E402
from homer.eval.nulls import _haar_rotation            # noqa: E402
cmp_mod = import_module("03_enigma_comparison")

ENIGMA_DIR = ROOT / "data_external" / "enigma"
GENE_DISORDERS = ["asd", "schizophrenia", "bipolar", "adhd"]   # in HOMER gene sets
HELDOUT = ["mdd", "ocd"]                                       # NOT in HOMER gene sets


def enigma_region_d(path):
    """{dk_region(lower, hemisphere-averaged): Cohen's d}."""
    df = pd.read_csv(path)
    acc = {}
    for _, row in df.iterrows():
        s = str(row["Structure"])
        if "_" not in s:
            continue
        reg = s.split("_", 1)[1].lower()
        d = row.get("d_icv")
        if pd.notna(d):
            acc.setdefault(reg, []).append(float(d))
    return {r: float(np.mean(v)) for r, v in acc.items()}


def main():
    print("=" * 78)
    print("ENIGMA — TRANSDIAGNOSTIC test (HOMER generic map vs shared cortical signature)")
    print("=" * 78)

    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    preds = dict(np.load(ROOT / "outputs/coupling/per_disorder_predictions.npz"))
    dk_centroids = cmp_mod.DK_REGIONS_MNI
    dk_to_parcels = cmp_mod.dk_to_homer_parcels(H.var, dk_centroids)
    dk_regions = list(dk_centroids.keys())

    # HOMER generic prediction = mean across disorders (they're ~identical)
    generic = np.mean([preds[d] for d in preds], axis=0)
    homer_dk = cmp_mod.aggregate_per_disorder_to_dk(generic, dk_to_parcels)
    homer_vec = np.array([homer_dk.get(r, np.nan) for r in dk_regions])

    # ENIGMA per-disorder + transdiagnostic average over the gene-set disorders
    fname = {"asd": "asd", "schizophrenia": "schizophrenia", "bipolar": "bipolar",
             "adhd": "adhd", "mdd": "mdd", "ocd": "ocd"}
    enigma = {}
    for key, stem in fname.items():
        p = ENIGMA_DIR / f"cortical_thickness_{stem}.csv"
        if p.exists():
            d = enigma_region_d(p)
            enigma[key] = np.array([d.get(r, np.nan) for r in dk_regions])
    trans_avg = np.nanmean(np.array([enigma[k] for k in GENE_DISORDERS if k in enigma]), axis=0)

    # DK centroids for the spin null (L centroid; hemisphere-collapsed map)
    cents = np.array([dk_centroids[r][:3] for r in dk_regions], float)

    def spin_p(a, b, n=5000, seed=0):
        c = cents - cents.mean(0)
        sph = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)
        m = np.isfinite(a) & np.isfinite(b)
        r_obs = pearsonr(a[m], b[m])[0]
        rng = np.random.default_rng(seed)
        null = np.empty(n)
        for t in range(n):
            _, perm = cKDTree(sph @ _haar_rotation(rng).T).query(sph)
            bp = b[perm]
            mm = np.isfinite(a) & np.isfinite(bp)
            null[t] = pearsonr(a[mm], bp[mm])[0]
        p = (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n + 1)
        return float(r_obs), float(p)

    print(f"\nHOMER generic predicted map vs ENIGMA observed (34 DK regions):")
    results = {}
    r_trans, p_trans = spin_p(homer_vec, trans_avg)
    print(f"  TRANSDIAGNOSTIC AVERAGE (ASD+SCZ+BD+ADHD): r = {r_trans:+.3f}  "
          f"spin p = {p_trans:.4f}  {'SURVIVES' if p_trans < 0.05 else 'n.s.'}")
    results["transdiagnostic_average"] = {"pearson_r": r_trans, "spin_p": p_trans}

    print(f"\n  per-disorder (HOMER generic vs each ENIGMA disorder):")
    for k in GENE_DISORDERS + HELDOUT:
        if k not in enigma:
            continue
        r, p = spin_p(homer_vec, enigma[k])
        tag = "(held-out, not in gene sets)" if k in HELDOUT else ""
        print(f"    {k:<15} r = {r:+.3f}  spin p = {p:.4f}  {tag}")
        results[k] = {"pearson_r": r, "spin_p": p, "in_gene_sets": k not in HELDOUT}

    out = {
        "dk_regions": dk_regions,
        "homer_generic_dk": homer_vec.tolist(),
        "enigma_transdiagnostic_dk": [float(x) for x in trans_avg],
        "results": results,
        "note": "HOMER generic = mean of per-disorder routed maps; transdiagnostic avg = "
                "mean ENIGMA Cohen's d across ASD/SCZ/BD/ADHD. Spin null over DK centroids.",
    }
    out_path = ROOT / "outputs" / "logs" / "enigma_transdiagnostic.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
