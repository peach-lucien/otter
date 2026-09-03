"""Integrity checks for the molecular and reverse-direction positive controls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs/logs"


def read(name: str) -> dict:
    return json.loads((LOGS / name).read_text())


def test_neuromaps_specificity_is_fully_logged() -> None:
    result = read("reverse_translation_neuromaps.json")
    summary = result["system_summary"]
    assert result["failures"] == []
    assert summary["dopamine"]["n_maps"] == 8
    assert summary["cannabinoid_cb1"]["n_maps"] == 1
    assert summary["gaba_a"]["n_maps"] == 2
    assert all("source_sha256" in row and "striatal_mass_fraction" in row
               for row in result["results"])
    assert result["specificity_test"]["u"] == 105.0
    assert result["specificity_test"]["p"] < 0.001
    assert summary["dopamine"]["median"] > summary["serotonin"]["median"]
    assert summary["dopamine"]["median"] > summary["cannabinoid_cb1"]["median"]
    assert summary["dopamine"]["median"] > summary["gaba_a"]["median"]


def test_reverse_direction_ed9_quantities_are_current_and_complete() -> None:
    result = read("reverse_translation_direction_diagnostic.json")
    uncovered = result["incoming_mass"]["fraction_below_0.1_uniform"]
    agreement = result["structure_agreement"]
    lowest = result["lowest_reverse_incoming"]
    classes = result["reverse_garin_classes"]
    assert result["forward_refit"]["entrywise_r_with_release"] == pytest.approx(1.0)
    assert result["forward_refit"]["argmax_match_with_release"] == 1.0
    assert uncovered["forward_human"] == pytest.approx(0.5042979943)
    assert uncovered["reverse_mouse"] == pytest.approx(0.3589055794)
    assert agreement["n_structures"] == 285
    assert agreement["median_r"] == pytest.approx(0.7435393858)
    assert lowest["n_eligible_structures"] == 148
    assert len(lowest["rows"]) == 12
    assert classes["mean_self_mass"] == pytest.approx(0.4310117377)
    assert len(classes["matrix_human_rows_mouse_columns"]) == 21
