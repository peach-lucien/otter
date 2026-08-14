"""Tests for otter.models, the four FGW model classes."""
import numpy as np
import pytest

from otter.models import (
    FGWModel,
    HierarchicalFGW,
    MultimodalFGW,
    SupervisedFGW,
    UnsupervisedGW,
)


# All four model classes share the FGWModel API, so most tests parametrise.
@pytest.fixture(params=[
    pytest.param(("UnsupervisedGW",   {"epsilon": 1e-2}),  id="UnsupervisedGW"),
    pytest.param(("SupervisedFGW",    {"epsilon": 1e-2}),  id="SupervisedFGW"),
    pytest.param(("MultimodalFGW",    {"epsilon": 1e-2, "use_sc": False}), id="MultimodalFGW_noSC"),
    pytest.param(("HierarchicalFGW",  {"epsilon": 1e-2}),  id="HierarchicalFGW"),
])
def model(request, mouse_ad, human_ad):
    """Yield a fitted instance of each model class."""
    cls_name, cfg = request.param
    cls_map = {
        "UnsupervisedGW": UnsupervisedGW,
        "SupervisedFGW":  SupervisedFGW,
        "MultimodalFGW":  MultimodalFGW,
        "HierarchicalFGW": HierarchicalFGW,
    }
    cls = cls_map[cls_name]
    m = cls(**cfg)
    m.fit(mouse_ad, human_ad)
    return m, cls_name


def test_pi_has_expected_shape(model, mouse_ad, human_ad):
    m, _ = model
    assert m.pi.shape == (mouse_ad.uns["n_nodes"], human_ad.uns["n_nodes"])


def test_pi_is_finite_and_nonneg(model):
    m, _ = model
    assert np.isfinite(m.pi).all()
    assert (m.pi >= 0).all()


def test_pi_marginal_for_semirelaxed(model):
    """Semirelaxed FGW: mouse marginal = 1/n_m. For UnsupervisedGW (balanced GW)
    rows sum to 1/n_m too. HierarchicalFGW assembles per-network so total = 1.
    """
    m, name = model
    n_m = m.pi.shape[0]
    if name == "HierarchicalFGW":
        # Block-sparse total mass should equal 1 (uniform over mouse)
        assert abs(m.pi.sum() - 1.0) < 1e-3
    else:
        # Each row sums to 1/n_m
        row_sums = m.pi.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0 / n_m, atol=1e-3)


def test_fit_info_populated(model):
    m, _ = model
    assert m.fit_info_ is not None
    assert isinstance(m.fit_info_.loss, float)


def test_repr_works_after_fit(model):
    m, _ = model
    s = repr(m)
    assert "shape" in s
    assert "loss=" in s


def test_evaluate_anchor(model):
    m, _ = model
    res = m.evaluate(eval_kind="anchor")
    # Full anchor recovery: should at least return finite top-1 in [0, 1]
    assert "top1" in res
    assert 0.0 <= res["top1"] <= 1.0


def test_evaluate_translation(model):
    m, _ = model
    res = m.evaluate(eval_kind="translation")
    assert "pearson_r_overall" in res


def test_predict_human_fc_shape(model, human_ad):
    m, _ = model
    fc_pred = m.predict_human_fc()
    assert fc_pred.shape == (human_ad.uns["n_nodes"],) * 2


def test_save_load_roundtrip(model, tmp_path, mouse_ad, human_ad):
    m, name = model
    p = tmp_path / "pi.npy"
    m.save(p)
    assert p.exists()
    assert (tmp_path / "pi.json").exists()

    # Load back
    cls_map = {
        "UnsupervisedGW": UnsupervisedGW,
        "SupervisedFGW":  SupervisedFGW,
        "MultimodalFGW":  MultimodalFGW,
        "HierarchicalFGW": HierarchicalFGW,
    }
    cls = cls_map[name]
    m_reloaded = cls.load(p)
    np.testing.assert_allclose(m.pi, m_reloaded.pi, atol=1e-5)
    assert m_reloaded.config == m.config


def test_unfit_model_raises_on_pi_access():
    m = SupervisedFGW()
    with pytest.raises(RuntimeError, match="not fit"):
        _ = m.pi


def test_supervised_holdout_anchor(mouse_ad, human_ad):
    """Withhold pair_id 1, the model still solves but those anchors aren't
    forced to align."""
    m = SupervisedFGW(epsilon=1e-2).fit(
        mouse_ad, human_ad, holdout_pair_ids=[1],
    )
    res = m.evaluate(held_out_pair_ids=[1], eval_kind="anchor")
    assert "top1" in res
    assert 0.0 <= res["top1"] <= 1.0


def test_multimodal_with_sc(mouse_ad, human_ad, synthetic_costs):
    m = MultimodalFGW(epsilon=1e-2, use_sc=True, sc_weight=0.3, fc_weight=0.7)
    m.fit(mouse_ad, human_ad,
            Cm_SC=synthetic_costs["Cm_SC"], Ch_SC=synthetic_costs["Ch_SC"])
    assert m.pi.shape == (20, 25)
    # Should report active modalities
    assert "weights" in m.fit_info_.extra
    assert "FC" in m.fit_info_.extra["weights"]
    assert "SC" in m.fit_info_.extra["weights"]


def test_multimodal_use_sc_without_costs_raises(mouse_ad, human_ad):
    m = MultimodalFGW(use_sc=True)
    with pytest.raises(ValueError, match="Cm_SC"):
        m.fit(mouse_ad, human_ad)


# ---------------------------------------------------------------------------
# Per-mouse-parcel xyz weighting (TOPO-1)


def test_multimodal_xyz_weight_per_parcel_changes_pi(mouse_ad, human_ad):
    """Zeroing the per-parcel xyz weight for grid rows changes π in those rows.

    Anchor rows (first 10 in the synthetic fixture) are dominated by anchor
    supervision so xyz contribution is overwritten; only grid rows feel xyz.
    """
    n_m = mouse_ad.uns["n_nodes"]
    n_h = human_ad.uns["n_nodes"]
    # Use a strong xyz_weight so the difference is visible at small scale
    m_baseline = MultimodalFGW(epsilon=1e-2, use_sc=False, xyz_weight=1.0)
    m_baseline.fit(mouse_ad, human_ad)
    pi_baseline = m_baseline.pi.copy()

    # Ablated: zero xyz for grid rows 12..16 (anchor rows are 0..9)
    w = np.full(n_m, 1.0)
    w[12:17] = 0.0
    m_ablated = MultimodalFGW(epsilon=1e-2, use_sc=False, xyz_weight=1.0)
    m_ablated.fit(mouse_ad, human_ad, xyz_weight_per_mouse_parcel=w)
    pi_ablated = m_ablated.pi.copy()

    assert pi_ablated.shape == (n_m, n_h)
    diff_ablated = np.abs(pi_baseline[12:17] - pi_ablated[12:17]).sum()
    assert diff_ablated > 1e-6, (
        f"Zeroing xyz for grid rows 12..17 should change those rows; "
        f"got total abs diff = {diff_ablated}"
    )


def test_multimodal_xyz_weight_per_parcel_matching_scalar_is_equivalent(
    mouse_ad, human_ad,
):
    """An array of all-0.5 should match the scalar xyz_weight=0.5 solve."""
    n_m = mouse_ad.uns["n_nodes"]
    w = np.full(n_m, 0.5)
    m_scalar = MultimodalFGW(epsilon=1e-2, use_sc=False, xyz_weight=0.5)
    m_scalar.fit(mouse_ad, human_ad)
    m_array  = MultimodalFGW(epsilon=1e-2, use_sc=False, xyz_weight=0.5)
    m_array.fit(mouse_ad, human_ad, xyz_weight_per_mouse_parcel=w)
    # Should converge to (approximately) the same π
    np.testing.assert_allclose(m_scalar.pi, m_array.pi, atol=1e-6)


def test_multimodal_xyz_weight_per_parcel_wrong_shape_raises(mouse_ad, human_ad):
    m = MultimodalFGW(epsilon=1e-2, use_sc=False)
    bad_w = np.zeros(5)   # wrong length
    with pytest.raises(ValueError, match="shape"):
        m.fit(mouse_ad, human_ad, xyz_weight_per_mouse_parcel=bad_w)
