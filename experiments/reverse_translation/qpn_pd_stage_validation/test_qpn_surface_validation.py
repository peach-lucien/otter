"""Outcome-free tests for fine-surface QPN validation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("qpn_surface_validation.py")
SPEC = importlib.util.spec_from_file_location("qpn_surface_validation", SCRIPT)
surface_analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = surface_analysis
SPEC.loader.exec_module(surface_analysis)


def test_subject_normalization_only_removes_formatting():
    assert surface_analysis.normalize_subject("sub-SYNTH123456") == "synth123456"
    assert surface_analysis.normalize_subject("SYNTH-123456") == "synth123456"


def test_aparc_parser_reads_thickness_column():
    payload = b"""# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd\nregion_a 2 3 4 2.25 0.1\nregion_b 4 5 6 1.75 0.2\n"""
    assert surface_analysis.parse_aparc_stats(payload) == {
        "region_a": 2.25,
        "region_b": 1.75,
    }


def test_local_destrieux_atlas_has_expected_regions():
    regions, atlas = surface_analysis.destrieux_labels(
        surface_analysis.ROOT / "data_external/nilearn"
    )
    assert len(regions) == 74
    assert len(np.unique(atlas.map_left)) == 75
    assert len(np.unique(atlas.map_right)) == 75
    assert "Medial_wall" not in regions


def test_all_archives_have_complete_nonidentifying_surface_matrix():
    archive_dir = surface_analysis.PROJECT_ROOT / "data_external/qpn-nc-r01"
    if not archive_dir.exists():
        pytest.skip("protected QPN archives are unavailable")
    regions, _ = surface_analysis.destrieux_labels(
        surface_analysis.ROOT / "data_external/nilearn"
    )
    frame, audit = surface_analysis.load_surface_thickness(
        archive_dir / "freesurfer_v7.3.2", regions
    )
    assert frame.shape == (271, 150)
    assert frame["_subject_key"].nunique() == 271
    assert audit["n_archives"] == 21
    assert audit["recon_all_success"] == 270
    assert audit["missing_parcel_values"] == 3
    assert audit["subjects_with_complete_148_region_surface"] == 270
    assert audit["subjects_eligible_after_surface_qc"] == 270


def test_surface_weight_mapping_and_small_spin_are_valid():
    regions, atlas = surface_analysis.destrieux_labels(
        surface_analysis.ROOT / "data_external/nilearn"
    )
    H, _ = surface_analysis.qpn.load_cached(
        "human", cache_dir=surface_analysis.ROOT / "outputs/anndata"
    )
    pi = np.load(surface_analysis.ROOT / "outputs/coupling/pi_canonical.npy")
    keys, weight, centroids, hemi_id, audit = surface_analysis.surface_geometry_and_weight(
        H, pi, surface_analysis.qpn.mouse_acronyms(), regions, atlas
    )
    assert len(keys) == len(weight) == len(centroids) == len(hemi_id) == 148
    assert np.isfinite(weight).all() and np.abs(weight).sum() > 0
    assert audit["mapped_OTTER_nodes"] == 1580
    rotations = surface_analysis.surface_rotations(centroids, hemi_id, 10, 123)
    assert rotations.shape == (10, 148)
    assert all(len(np.unique(rotation)) == 148 for rotation in rotations)


def test_rank_residuals_remove_monotone_covariate():
    rng = np.random.default_rng(5)
    covariate = np.arange(200, dtype=float)
    values = covariate + rng.normal(0, 10, len(covariate))
    residual = surface_analysis.rank_residuals(values, covariate[:, None])
    assert abs(np.corrcoef(residual, covariate)[0, 1]) < 1e-10
