"""ROADMAP item C: confidence-weighted FC using fc_n_obs.

A structural diagnostic, not a CV experiment. We check how much per-cell
shrinkage by fc_n_obs would change the resulting FC cost matrix C. If it
barely moves the matrix, no CV experiment is warranted.

Saves outputs/logs/confidence_weighted_fc_check.json.

Findings (2026-05-04):
    Mouse fc_n_obs uniformly = 105 → confidence weighting is a literal no-op.
    Human fc_n_obs: min 100, max 113, 85% of cells at max, all 42 anchor
        nodes at row_cov 99.8%. Shrunk-vs-original C correlate at r=0.99997,
        |C_shrunk - C_orig|_max = 0.045 on a [0, 2] scale.
    → no CV experiment warranted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached                          # noqa: E402
from homer.data.anchors import get_anchor_index             # noqa: E402
from homer.costs import correlation_distance                # noqa: E402

ANN = ROOT / "outputs" / "anndata"
LOG = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)


def per_species_check(species: str) -> dict:
    ad, _ = load_cached(species, cache_dir=ANN)
    fc = np.asarray(ad.uns["fc_mean"]).astype(np.float64)
    n_obs = np.asarray(ad.uns["fc_n_obs"]).astype(np.float64)
    n_max = float(n_obs.max())

    od_mask = ~np.eye(n_obs.shape[0], dtype=bool)
    n_off = n_obs[od_mask]
    row_cov = n_obs.mean(axis=1) / n_max

    # Bayesian-flavored shrinkage: pull r toward 0 proportional to coverage deficit
    fc_shrunk = fc * (n_obs / n_max)

    C_orig   = correlation_distance(fc)
    C_shrunk = correlation_distance(fc_shrunk)

    ut = np.triu_indices_from(C_orig, k=1)
    corr = float(np.corrcoef(C_orig[ut], C_shrunk[ut])[0, 1])

    idx = get_anchor_index(ad.var)
    anchor_cov = row_cov[idx.pos]

    return {
        "species":            species,
        "n_obs_min":          float(n_off.min()),
        "n_obs_max":          float(n_off.max()),
        "n_obs_mean":         float(n_off.mean()),
        "frac_cells_at_max":  float((n_off == n_off.max()).mean()),
        "row_cov_min":        float(row_cov.min()),
        "row_cov_median":     float(np.median(row_cov)),
        "anchor_row_cov_min": float(anchor_cov.min()),
        "anchor_row_cov_max": float(anchor_cov.max()),
        "C_corr_orig_shrunk": corr,
        "C_diff_max":         float(np.abs(C_orig - C_shrunk).max()),
        "C_diff_mean":        float(np.abs(C_orig - C_shrunk)[od_mask].mean()),
    }


def main():
    out = {sp: per_species_check(sp) for sp in ["mouse", "human"]}
    print(json.dumps(out, indent=2))
    LOG.joinpath("confidence_weighted_fc_check.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {LOG / 'confidence_weighted_fc_check.json'}")
    print("\nVERDICT: confidence weighting via fc_n_obs is structurally a no-op")
    print("         because input FC matrices have near-uniform coverage.")


if __name__ == "__main__":
    main()
