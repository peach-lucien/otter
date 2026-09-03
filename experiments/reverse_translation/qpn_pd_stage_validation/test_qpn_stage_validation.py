"""Outcome-free tests for the public QPN-NC stage-validation workflow."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).with_name("qpn_stage_validation.py")
SPEC = importlib.util.spec_from_file_location("qpn_stage_validation", SCRIPT)
qpn = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = qpn
SPEC.loader.exec_module(qpn)


def test_closest_assessment_uses_distance_then_baseline():
    frame = pd.DataFrame(
        {
            "participant_id": ["a", "a", "b", "b"],
            "redcap_event_name": ["12 Months", "Baseline", "Baseline", "12 Months"],
            "score": [4, 3, 8, 9],
            "lag": [-0.2, 0.2, 2.0, 0.1],
        }
    )
    selected = qpn.closest_assessment(frame, "score", "lag").set_index("participant_id")
    assert selected.loc["a", "score"] == 3
    assert selected.loc["b", "score"] == 9


def synthetic_controls(n: int = 48) -> pd.DataFrame:
    age = np.linspace(45, 80, n)
    sex = np.where(np.arange(n) % 2, "Male", "Female")
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "MRI_age": age,
            "sex": sex,
            "r1": 3.2 - 0.006 * age + 0.03 * (sex == "Male") + rng.normal(0, 0.02, n),
            "r2": 2.9 - 0.003 * age - 0.02 * (sex == "Male") + rng.normal(0, 0.02, n),
            "r3": 3.0 - 0.004 * age + rng.normal(0, 0.02, n),
        }
    )


def test_control_normative_atrophy_has_correct_sign():
    controls = synthetic_controls()
    model = qpn.fit_normative(controls, ["r1", "r2", "r3"])
    case = pd.DataFrame(
        {"MRI_age": [65], "sex": ["Female"], "r1": [2.3], "r2": [2.7], "r3": [2.6]}
    )
    atrophy = qpn.control_normative_atrophy(case, ["r1", "r2", "r3"], model)
    assert atrophy.shape == (1, 3)
    assert atrophy[0, 0] > atrophy[0, 1]
    assert np.all(atrophy > 0)


def test_severity_slope_recovers_regional_direction_with_covariates():
    rng = np.random.default_rng(7)
    n = 80
    stage = np.repeat([1.0, 2.0, 3.0, 4.0], n // 4)
    frame = pd.DataFrame(
        {
            "hy": stage,
            "MRI_age": 55 + 4 * stage + rng.normal(0, 2, n),
            "sex": np.where(np.arange(n) % 2, "Male", "Female"),
            "lag": rng.normal(0, 1, n),
        }
    )
    maps = np.column_stack(
        [1.2 * stage, -0.8 * stage, 0.1 * stage]
    ) + rng.normal(0, 0.2, (n, 3))
    slope = qpn.severity_slope_map(maps, frame, "hy", "lag")
    assert slope[0] > 0.9
    assert slope[1] < -0.5
    assert abs(slope[2]) < 0.3


def test_spatial_summary_matches_explicit_rotation():
    slope = np.array([2.0, 0.5, -1.0, -2.0])
    weight = np.array([0.4, -0.2, 0.1, -0.3])
    keys = [("L", "a"), ("L", "b"), ("R", "a"), ("R", "b")]
    rotations = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=int)
    observed = qpn.spatial_summary(slope, weight, rotations, keys)
    z = qpn.zscore_map(slope)
    assert observed["D"] == pytest.approx(weight @ z)
    explicit = np.array([weight @ z[p] for p in rotations])
    p_one = (np.sum(explicit >= observed["D"]) + 1) / 3
    assert observed["spatial_p_one_sided"] == pytest.approx(p_one)
    p_lower = (np.sum(explicit <= observed["D"]) + 1) / 3
    assert observed["spatial_p_two_sided"] == pytest.approx(min(1, 2 * min(p_one, p_lower)))


def test_unscaled_spatial_map_is_still_mean_centered():
    slope = np.array([10.0, 11.0, 12.0, 13.0])
    weight = np.array([1.0, 1.0, 1.0, 1.0])
    keys = [("L", "a"), ("L", "b"), ("R", "a"), ("R", "b")]
    rotations = np.array([[0, 1, 2, 3]], dtype=int)
    observed = qpn.spatial_summary(slope, weight, rotations, keys, scaled=False)
    assert observed["D"] == pytest.approx(0.0)


def test_bootstrap_resamples_complete_normative_and_severity_pipeline():
    controls = synthetic_controls(48)
    rng = np.random.default_rng(12)
    n = 64
    stage = np.repeat([1.0, 2.0, 3.0, 4.0], n // 4)
    patients = pd.DataFrame(
        {
            "MRI_age": 58 + 3 * stage + rng.normal(0, 2, n),
            "sex": np.where(np.arange(n) % 2, "Male", "Female"),
            "hy": stage,
            "lag": rng.normal(0, 1, n),
            "r1": 2.9 - 0.03 * stage + rng.normal(0, 0.02, n),
            "r2": 2.7 + 0.02 * stage + rng.normal(0, 0.02, n),
            "r3": 2.8 - 0.01 * stage + rng.normal(0, 0.02, n),
        }
    )
    draws = qpn.bootstrap_D(
        controls,
        patients,
        ["r1", "r2", "r3"],
        "hy",
        "lag",
        np.array([0.4, -0.2, 0.1]),
        25,
        8,
    )
    assert draws.shape == (25,)
    assert np.isfinite(draws).all()


def test_protected_release_metadata_and_month_units():
    archive_dir = qpn.PROJECT_ROOT / "data_external/qpn-nc-r01"
    if not archive_dir.exists():
        pytest.skip("protected QPN archives are not locally available")
    controls, patients, loaded = qpn.load_qpn(archive_dir)
    assert len(controls) == 69
    assert len(patients) == 141
    assert loaded["audit"]["n_within_one_month"] == 117
    assert loaded["audit"]["small_stage_cells_suppressed"] is True
    assert "hy_counts_raw" not in loaded["audit"]
    assert patients["hy_ordinal"].max() == 4
    assert np.isfinite(patients["hy_lag_months"]).all()


def test_partial_spearman_removes_ranked_covariate():
    rng = np.random.default_rng(14)
    covariate = np.arange(200, dtype=float)
    x = covariate + rng.normal(0, 10, len(covariate))
    y = covariate + rng.normal(0, 10, len(covariate))
    assert abs(qpn.partial_spearman(x, y, covariate[:, None])) < 0.2


def test_committed_qpn_outputs_are_aggregate_only():
    logs = qpn.ROOT / "outputs/logs"
    paths = [
        logs / "reverse_translation_qpn_pd_stage.json",
        logs / "reverse_translation_qpn_pd_stage_surface.json",
    ]
    forbidden_keys = {
        "participant_id",
        "subject_key",
        "participant_scores",
        "participant_maps",
        "stage_slope_z",
        "region_order",
        "archive_files",
        "archive_sizes",
    }

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key.casefold().lstrip("_")
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    for path in paths:
        result = json.loads(path.read_text())
        assert forbidden_keys.isdisjoint(keys(result))
        assert re.search(r"(?i)(?:sub[-_])?qpn[-_]?nc[-_]?\d{3,}", path.read_text()) is None
        assert result["privacy"]["participant_identifiers_written"] is False
        assert result["privacy"]["participant_maps_or_scores_written"] is False


def test_release_archive_excludes_restricted_qpn_inputs():
    script = (qpn.ROOT / "scripts/build_archives.sh").read_text()
    assert "--exclude='data_external/qpn-nc-r*'" in script
    assert "--exclude='data_external/**/qpn-nc-r*'" in script
