"""Shared synthetic fixture for the test suite.

A tiny, fast-to-fit cross-species problem: 20 mouse nodes, 25 human nodes,
5 anchor pair_ids × 2 hemispheres = 10 anchors per species. Every test fits
in <1 second.

Why synthetic: avoids needing the ~100 MB real h5ad cache. Tests can run
on a fresh checkout without external data.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Make `src/homer/` importable from a fresh checkout without requiring
# `pip install -e .` or `PYTHONPATH=src`. This conftest is loaded by pytest
# before any test module, so the sys.path injection happens early enough.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import anndata as ad
import numpy as np
import pandas as pd
import pytest

# Suppress the deprecation warnings from the shim modules during tests so
# pytest output stays readable. Tests themselves can re-enable if needed.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="homer\\..*")


N_MOUSE  = 20
N_HUMAN  = 25
N_PAIRS  = 5     # 5 pair_ids × 2 hemispheres = 10 anchors per species


def _build_var(n_nodes: int, n_anchors: int, *, seed: int) -> pd.DataFrame:
    """Build a synthetic adata.var with n_anchors anchors split across 5 networks.

    Anchors get pair_ids 1..N_PAIRS (×2 hemispheres). The first 5 networks of
    `homer.data.networks.NETWORKS` are used (one per pair_id).
    """
    from homer.data.networks import NETWORKS, PAIRID_TO_NETWORK
    rng = np.random.default_rng(seed)
    types = np.zeros(n_nodes, dtype=np.int8)
    pairids = np.zeros(n_nodes, dtype=np.int32)
    regions = []
    hemis = []

    # Anchors first (alternating L/R for each pair_id)
    pair_ids_to_use = sorted(PAIRID_TO_NETWORK.keys())[:N_PAIRS]
    for i in range(n_anchors):
        pid = pair_ids_to_use[i // 2]
        hemi = "L" if (i % 2 == 0) else "R"
        types[i] = 1
        pairids[i] = pid
        regions.append(f"{hemi}_test_anchor_{pid}")
        hemis.append(hemi)
    # Grid nodes, random pair_ids, alternating hemis
    for i in range(n_anchors, n_nodes):
        pid = int(rng.integers(N_PAIRS + 1, 12))   # non-anchor pair_id
        hemi = "L" if (i % 2 == 0) else "R"
        types[i] = 2
        pairids[i] = pid
        regions.append(f"{hemi}_grid_{i}")
        hemis.append(hemi)

    coords = rng.uniform(-1, 1, size=(n_nodes, 3))
    df = pd.DataFrame({
        "type":      types,
        "numid":     np.arange(1, n_nodes + 1, dtype=np.int32),
        "pairid":    pairids,
        "region":    regions,
        "subregion": ["" for _ in range(n_nodes)],
        "x":         coords[:, 0],
        "y":         coords[:, 1],
        "z":         coords[:, 2],
        "hemisphere": hemis,
        "garin_anchor": types == 1,
        "anchor_pair_id": pd.array([p if t == 1 else pd.NA for p, t in zip(pairids, types)],
                                     dtype="Int64"),
    })
    df.index = df["numid"].astype(int).astype(str)
    df.index.name = "node_id"
    return df


def _build_synthetic_ad(n_nodes: int, n_anchors: int, *, n_subjects: int,
                         seed: int) -> "ad.AnnData":
    """Build a tiny AnnData with a random symmetric FC matrix in uns."""
    rng = np.random.default_rng(seed)
    var = _build_var(n_nodes, n_anchors, seed=seed)
    fc = rng.uniform(-0.5, 0.5, size=(n_nodes, n_nodes)).astype(np.float32)
    fc = 0.5 * (fc + fc.T)
    np.fill_diagonal(fc, 1.0)
    n_obs = np.full((n_nodes, n_nodes), n_subjects, dtype=np.int32)
    A = ad.AnnData(
        X=np.zeros((n_subjects, n_nodes), dtype=np.float32),
        var=var,
        obs=pd.DataFrame({"subject_id": [f"sub_{i:03d}" for i in range(n_subjects)]})
            .set_index("subject_id"),
        uns={
            "species":   "synthetic",
            "fc_mean":   fc,
            "fc_n_obs":  n_obs,
            "n_nodes":   n_nodes,
            "n_subjects": n_subjects,
        },
    )
    return A


@pytest.fixture(scope="session")
def mouse_ad():
    """Synthetic mouse AnnData (20 nodes, 10 anchors, 8 subjects)."""
    return _build_synthetic_ad(N_MOUSE, n_anchors=N_PAIRS * 2, n_subjects=8, seed=0)


@pytest.fixture(scope="session")
def human_ad():
    """Synthetic human AnnData (25 nodes, 10 anchors, 10 subjects)."""
    return _build_synthetic_ad(N_HUMAN, n_anchors=N_PAIRS * 2, n_subjects=10, seed=1)


@pytest.fixture(scope="session")
def synthetic_costs(mouse_ad, human_ad):
    """Pre-built FC + SC cost matrices for the synthetic data."""
    from homer.costs import correlation_distance, normalise_cost
    fc_m = mouse_ad.uns["fc_mean"].astype(np.float64)
    fc_h = human_ad.uns["fc_mean"].astype(np.float64)
    Cm_FC = normalise_cost(correlation_distance(fc_m), scheme="max")
    Ch_FC = normalise_cost(correlation_distance(fc_h), scheme="max")

    rng = np.random.default_rng(42)
    Cm_SC = rng.uniform(0, 1, size=(N_MOUSE, N_MOUSE))
    Cm_SC = 0.5 * (Cm_SC + Cm_SC.T); np.fill_diagonal(Cm_SC, 0.0)
    Ch_SC = rng.uniform(0, 1, size=(N_HUMAN, N_HUMAN))
    Ch_SC = 0.5 * (Ch_SC + Ch_SC.T); np.fill_diagonal(Ch_SC, 0.0)

    return {"Cm_FC": Cm_FC, "Ch_FC": Ch_FC, "Cm_SC": Cm_SC, "Ch_SC": Ch_SC}
