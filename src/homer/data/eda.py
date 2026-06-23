"""Exploratory analysis helpers, streaming-friendly so they fit on the sandbox.

Lightweight functions used by the EDA notebook + FC-translation evaluation. The
main job is to compute things over the per-subject FC tensor *without* ever
materialising it in memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import spearmanr

from homer.data.io import _MAT_TOPKEY, _mat_path


# ---------------------------------------------------------------------------
# Streaming subject-to-subject FC similarity
# ---------------------------------------------------------------------------
def streaming_subject_similarity(
    species: str,
    *,
    data_dir: Path | None = None,
) -> np.ndarray:
    """(n_subj × n_subj) Pearson correlation between subjects' vectorised FC matrices."""
    p = _mat_path(species, data_dir)
    top = _MAT_TOPKEY[species]
    with h5py.File(str(p), "r") as f:
        rr = f[f"{top}/rr"]
        n_subj, n_nodes, _ = rr.shape
        block = rr.chunks[1] if rr.chunks else 256

        sum_  = np.zeros(n_subj, dtype=np.float64)
        sum2_ = np.zeros(n_subj, dtype=np.float64)
        sp_   = np.zeros((n_subj, n_subj), dtype=np.float64)
        n_features = 0

        for b in range((n_nodes + block - 1) // block):
            j0, j1 = b * block, min((b + 1) * block, n_nodes)
            cd = rr[:, :, j0:j1]
            X = cd.reshape(n_subj, -1).astype(np.float32, copy=False)
            np.nan_to_num(X, copy=False)
            sum_  += X.sum(axis=1, dtype=np.float64)
            sum2_ += (X.astype(np.float64) ** 2).sum(axis=1)
            sp_   += (X @ X.T).astype(np.float64)
            n_features += X.shape[1]

    n = float(n_features)
    mu = sum_ / n
    var = sum2_ / n - mu * mu
    cov = sp_ / n - np.outer(mu, mu)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.sqrt(np.outer(var.clip(min=1e-12), var.clip(min=1e-12)))
        corr = cov / denom
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


# ---------------------------------------------------------------------------
# Anchor-anchor cross-species Spearman correlation
# ---------------------------------------------------------------------------
def anchor_submatrix(
    fc_mean: np.ndarray,
    var: Any,
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Return the 42×42 inter-anchor FC submatrix and the ordered (pair_id, hemi) list."""
    anchors = (
        var.loc[var["garin_anchor"]]
        .reset_index()
        .sort_values(["anchor_pair_id", "hemisphere"], kind="stable")
    )
    pos = anchors["numid"].astype(int).values - 1
    sub = fc_mean[np.ix_(pos, pos)]
    pairs = list(zip(anchors["anchor_pair_id"].astype(int), anchors["hemisphere"]))
    return sub, pairs


def cross_species_anchor_spearman(
    fc_h: np.ndarray, var_h, fc_m: np.ndarray, var_m,
) -> dict[str, float]:
    """Spearman correlation between human and mouse 42×42 inter-anchor FC matrices."""
    sub_h, pairs_h = anchor_submatrix(fc_h, var_h)
    sub_m, pairs_m = anchor_submatrix(fc_m, var_m)
    if pairs_h != pairs_m:
        raise ValueError(
            f"anchor ordering mismatch, human={pairs_h[:3]} mouse={pairs_m[:3]}"
        )
    n = sub_h.shape[0]
    iu = np.triu_indices(n, k=1)
    vh = sub_h[iu]
    vm = sub_m[iu]
    valid = np.isfinite(vh) & np.isfinite(vm)
    rho, p = spearmanr(vh[valid], vm[valid])
    pearson = float(np.corrcoef(vh[valid], vm[valid])[0, 1])
    return {
        "spearman_rho": float(rho),
        "spearman_p":   float(p),
        "pearson_r":    pearson,
        "n_pairs_used": int(valid.sum()),
        "n_anchors":    int(n),
    }


# ---------------------------------------------------------------------------
# Within-species L/R hemispheric symmetry per pair
# ---------------------------------------------------------------------------
def lr_homologue_correlation(fc_mean: np.ndarray, var) -> dict[str, Any]:
    """For each pair_id, correlate the FC fingerprint of the L and R partner."""
    out_rho = []
    out_pid = []
    pids = sorted(var["pairid"].unique().astype(int))
    for pid in pids:
        rows = var[var["pairid"] == pid]
        if len(rows) != 2:
            continue
        l = rows[rows["hemisphere"] == "L"]
        r = rows[rows["hemisphere"] == "R"]
        if len(l) != 1 or len(r) != 1:
            continue
        i_l = int(l["numid"].iloc[0]) - 1
        i_r = int(r["numid"].iloc[0]) - 1
        v_l = fc_mean[i_l]; v_r = fc_mean[i_r]
        valid = np.isfinite(v_l) & np.isfinite(v_r)
        if valid.sum() < 10:
            continue
        rho = float(np.corrcoef(v_l[valid], v_r[valid])[0, 1])
        out_rho.append(rho)
        out_pid.append(int(pid))
    rhos = np.asarray(out_rho)
    return {
        "pair_ids":     out_pid,
        "rhos":         rhos.tolist(),
        "median_rho":   float(np.median(rhos)),
        "mean_rho":     float(np.mean(rhos)),
        "min_rho":      float(np.min(rhos)),
        "frac_above_05": float((rhos > 0.5).mean()),
    }
