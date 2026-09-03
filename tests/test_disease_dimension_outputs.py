"""Integrity checks for the committed clinical disease-dimension results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs/logs"


def load_strict(name: str) -> dict:
    def reject_nonfinite(value: str):
        raise ValueError(f"non-finite JSON constant {value} in {name}")

    return json.loads((LOGS / name).read_text(), parse_constant=reject_nonfinite)


def all_positive(result: dict) -> bool:
    return all(value > 0 for value in result["row_selectivity"].values())


def test_alzheimer_discovery_and_external_confirmation() -> None:
    discovery = load_strict("reverse_translation_ad_phenotypes.json")
    for coupling in ("canonical", "no_relevant_anchor_packs"):
        results = discovery["results"][coupling]["parcel_balanced"]
        for modality in ("tau", "vbm"):
            assert all_positive(results[modality])
            assert results[modality]["joint_selectivity"] > 0
            assert results[modality]["p_one_sided"] < 0.05

    confirmation = load_strict("reverse_translation_ad_leads_confirmation.json")
    assert confirmation["confirmation_gate_passed"] is True
    for coupling in ("canonical", "no_relevant_anchor_packs"):
        result = confirmation["results"][coupling]["parcel_balanced"][
            "primary_thresholded_subtype_vs_rest"
        ]
        assert all_positive(result)
        assert result["joint_selectivity"] > 0
        assert result["p_one_sided"] < 0.05


def test_tms_primary_and_pack_removed_results() -> None:
    result = load_strict("reverse_translation_symptom_dissociation.json")
    assert result["schema_version"] == "3.0.0"
    headline = result["headline"]
    assert headline["primary_weighting"] == "parcel_mass"
    assert headline["canonical_primary_top_10"]["contrast_C"] > 0
    assert headline["canonical_primary_top_10"]["p_max_fwer"] < 0.05
    assert headline["pack_out_primary_top_10"]["contrast_C"] > 0
    assert headline["pack_out_primary_top_10"]["p_max_fwer"] < 0.05
    assert headline["paired_primary_top_10"]["delta_C"] > 0
    assert headline["paired_primary_top_10"]["p_max_fwer"] < 0.05


def test_parkinson_stage_and_qpn_validation() -> None:
    enigma = load_strict("reverse_translation_pd_stage_progression.json")
    canonical = enigma["couplings"]["pi_canonical.npy"]
    assert canonical["stage_bias"]["HY1"] < 0
    assert canonical["stage_bias"]["HY45"] > 0
    assert canonical["linear_trend"] > 0
    assert canonical["p_two_sided"] < 0.05

    qpn = load_strict("reverse_translation_qpn_pd_stage.json")
    assert qpn["stage_formulations"]["participant_partial_spearman"][
        "spatial_p_one_sided"
    ] < 0.05
    assert qpn["stage_formulations"]["HY4_5_Huber"]["spatial_p_one_sided"] < 0.05
    assert qpn["synchronized_max_statistic"]["spatial_p_one_sided"] < 0.05
    assert qpn["HY4_5_OLS"]["influence"]["leave_one_pd"]["all_positive"] is True
    assert qpn["HY4_5_OLS"]["influence"]["leave_one_control"]["all_positive"] is True
    assert qpn["privacy"]["participant_identifiers_written"] is False

    surface = load_strict("reverse_translation_qpn_pd_stage_surface.json")
    assert surface["primary_HY_surface"]["D"] > 0
    assert all(value > 0 for value in surface["primary_HY_surface"]["hemisphere_D"].values())
    assert surface["primary_HY_surface"]["influence"]["leave_one_pd"][
        "all_positive"
    ] is True
    assert surface["privacy"]["participant_maps_or_scores_written"] is False
