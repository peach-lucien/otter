"""Tests for the data layer.

These tests must pass before any modeling work proceeds. They check:
    - shape & cardinality of each species' data
    - the 42 anchor / 21 pairid invariants are satisfied and consistent across species
    - mean FC is symmetric with unit diagonal
    - hemisphere flag matches the L_/R_ prefix and pairs are L+R balanced
    - subjects have non-trivial FC (sanity: not all zeros, not all NaN)
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from otter.data import DATA_DIR, build_anndata, parse_t_table, load_cached, load_struct


# Fixtures ---------------------------------------------------------------------
# Use cached AnnData if available; .mat loading + nanmean is ~30 s/species.
CACHE = Path(__file__).resolve().parents[1] / "outputs" / "anndata"


@pytest.fixture(scope="session")
def adatas() -> dict[str, ad.AnnData]:
    out = {}
    for sp in ("human", "mouse"):
        h5 = CACHE / f"{sp}.h5ad"
        if h5.exists():
            A, _ = load_cached(sp, cache_dir=CACHE)
            out[sp] = A
        else:
            try:
                out[sp] = build_anndata(sp, cache_dir=CACHE, overwrite=True)
            except FileNotFoundError:
                pytest.skip(
                    f"{sp}: no cached .h5ad and no raw .mat source available. "
                    "Run `python scripts/fetch_data.py` for the caches, or set "
                    "DATA_DIR to the raw source to rebuild."
                )
    return out


# Core tests ------------------------------------------------------
def test_shapes(adatas):
    H, M = adatas["human"], adatas["mouse"]
    assert H.uns["fc_mean"].shape == (2094, 2094)
    assert M.uns["fc_mean"].shape == (1864, 1864)
    assert H.uns["n_subjects"] == 113
    assert M.uns["n_subjects"] == 105
    assert (H.var["garin_anchor"]).sum() == 42
    assert (M.var["garin_anchor"]).sum() == 42


def test_anchor_pairing_consistent(adatas):
    H, M = adatas["human"], adatas["mouse"]
    h_anchor_pairs = set(H.var.loc[H.var["garin_anchor"], "anchor_pair_id"].dropna().astype(int))
    m_anchor_pairs = set(M.var.loc[M.var["garin_anchor"], "anchor_pair_id"].dropna().astype(int))
    # 21 paired regions (Garin atlas), with both L and R present in each species
    assert h_anchor_pairs == m_anchor_pairs == set(range(1, 22)), (
        f"anchor pair ids differ, human={sorted(h_anchor_pairs)} "
        f"mouse={sorted(m_anchor_pairs)}"
    )
    # Each pair id should appear exactly twice (once L, once R) in each species' anchors
    for sp_name, A in (("human", H), ("mouse", M)):
        anchors = A.var.loc[A.var["garin_anchor"]]
        counts = anchors.groupby("anchor_pair_id", observed=True).size()
        assert (counts == 2).all(), f"{sp_name} anchor pair counts != 2: {counts.to_dict()}"
        hemis = anchors.groupby("anchor_pair_id", observed=True)["hemisphere"].nunique()
        assert (hemis == 2).all(), f"{sp_name} anchor pairs missing L/R: {hemis.to_dict()}"


def test_fc_symmetric_unit_diag(adatas):
    """Per-subject FC is symmetric by construction; nanmean preserves symmetry only
    when both (i,j) and (j,i) are NaN/non-NaN together. Test on non-NaN entries only,
    after asserting NaN occurs at most where fc_n_obs == 0.
    """
    for sp, A in adatas.items():
        fc = A.uns["fc_mean"]
        n_obs = A.uns["fc_n_obs"]
        # NaNs in fc_mean must coincide with zero-coverage entries
        if np.isnan(fc).any():
            assert (n_obs[np.isnan(fc)] == 0).all(), (
                f"{sp}: NaN in fc_mean where n_obs>0"
            )
        valid = ~np.isnan(fc) & ~np.isnan(fc.T)
        diff = np.abs(fc - fc.T)
        max_asym = float(diff[valid].max()) if valid.any() else 0.0
        assert max_asym <= 1e-5, f"{sp}: fc_mean asymmetric, max diff={max_asym:.2e}"
        diag = np.diag(fc)
        valid_diag = ~np.isnan(diag)
        assert valid_diag.any(), f"{sp}: entire diagonal NaN"
        assert np.allclose(diag[valid_diag], 1.0, atol=1e-3), (
            f"{sp}: fc_mean diagonal not unit (mean={diag[valid_diag].mean():.4f})"
        )


# Additional sanity tests ------------------------------------------------------
def test_hemisphere_flag_matches_prefix(adatas):
    for sp, A in adatas.items():
        v = A.var
        l_mask = v["region"].str.startswith("L_")
        r_mask = v["region"].str.startswith("R_")
        assert (l_mask | r_mask).all(), f"{sp}: nodes lacking L_/R_ prefix"
        assert (v.loc[l_mask, "hemisphere"] == "L").all()
        assert (v.loc[r_mask, "hemisphere"] == "R").all()
        # within-species L/R balance
        assert l_mask.sum() == r_mask.sum() == len(v) // 2


def test_pairid_within_species_pairs_lr(adatas):
    """Within a species, every pairid should occur exactly once L and once R."""
    for sp, A in adatas.items():
        v = A.var
        counts = v.groupby("pairid")["hemisphere"].agg(set)
        bad = counts[counts != {"L", "R"}]
        assert bad.empty, f"{sp}: pairids missing both hemispheres: {bad.to_dict()}"


def test_fc_finite_and_in_range(adatas):
    for sp, A in adatas.items():
        fc = A.uns["fc_mean"]
        # We allow NaN where coverage is zero; everything else must be finite.
        valid = ~np.isnan(fc)
        assert np.isfinite(fc[valid]).all(), f"{sp}: Inf in fc_mean"
        off_diag_mask = valid & ~np.eye(fc.shape[0], dtype=bool)
        vals = fc[off_diag_mask]
        assert vals.min() >= -1.0 - 1e-5
        assert vals.max() <=  1.0 + 1e-5
        # NaN budget, sanity report (not a hard fail)
        nan_pct = float(np.isnan(fc).mean() * 100)
        assert nan_pct <= 2.0, f"{sp}: surprising NaN rate in fc_mean: {nan_pct:.2f}%"


def test_node_index_consistency(adatas):
    """numid should be 1..n_nodes contiguous, and the var index should match."""
    for sp, A in adatas.items():
        n = A.uns["n_nodes"]
        np.testing.assert_array_equal(A.var["numid"].values, np.arange(1, n + 1))
        assert list(A.var.index) == [str(i) for i in range(1, n + 1)]


def test_voxel_indices_present_and_nonempty():
    """Voxel index lists are stripped before .h5ad write (anndata can't serialise
    ragged object arrays). Test against a fresh build instead.
    """
    from otter.data import load_metadata, parse_t_table
    for sp in ("mouse",):  # mouse is small/fast; sufficient as a smoke test
        try:
            meta = load_metadata(sp)
        except FileNotFoundError:
            pytest.skip(
                "raw .mat source (DATA_DIR) not present; rebuilding metadata from "
                "source is maintainer-only. Downstream users get the built caches "
                "via `python scripts/fetch_data.py`."
            )
        df = parse_t_table(meta["t"], meta["ht"])
        vox = df["voxel_indices"].tolist()
        assert len(vox) == 1864
        assert all(len(v) > 0 for v in vox), f"{sp}: empty voxel list for some node"


# Light unit tests on parse_t_table without full mat load ----------------------
def test_parse_t_table_with_synthetic():
    ht = ["type", "numid", "pairid", "region", "subregion", "center", "indices"]
    t = [
        [np.array(1.0), np.array(1.0), np.array(1.0), "L_RegionA", "subA",
         np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0])],
        [np.array(1.0), np.array(2.0), np.array(1.0), "R_RegionA", "subA",
         np.array([-1.0, 2.0, 3.0]), np.array([11.0, 21.0])],
        [np.array(2.0), np.array(3.0), np.array(2.0), "L_Grid_1", "",
         np.array([0.0, 0.0, 0.0]), np.array([100.0])],
    ]
    df = parse_t_table(t, ht)
    assert df.shape[0] == 3
    assert df.iloc[0]["hemisphere"] == "L"
    assert bool(df.iloc[0]["garin_anchor"]) is True
    assert bool(df.iloc[2]["garin_anchor"]) is False
    assert int(df.iloc[0]["anchor_pair_id"]) == 1
    assert pd.isna(df.iloc[2]["anchor_pair_id"])
    assert df.iloc[2]["x"] == 0.0


# Smoke test: data dir exists --------------------------------------------------
def test_data_dir_present():
    if not DATA_DIR.exists():
        pytest.skip(
            f"DATA_DIR (raw .mat source) not present: {DATA_DIR}. Only needed to "
            "rebuild the caches from source; downstream users fetch the built "
            "caches via `python scripts/fetch_data.py`."
        )
    assert (DATA_DIR / "corrs_human.mat").exists()
    assert (DATA_DIR / "corrs_mouse.mat").exists()
