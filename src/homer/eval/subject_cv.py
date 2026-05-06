"""Subject-level K-fold CV.

Independent of held-out anchor CV (which tests SPATIAL generalisation). This
tests SUBJECT generalisation: does π trained on 80% of subjects predict the
held-out 20%'s mean FC?

Public API:
    subject_kfold_cv(model_factory, *, k_folds=5, test_frac=0.20, seed=0)
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from homer.data.anchors import get_anchor_index
from homer.data.io import load_cached, stream_mean_fc_subset
from homer.data.networks import assign_networks
from homer.eval.translation import fc_translation_quality


def subject_kfold_cv(
    model_factory: Callable,
    cache_dir,
    *,
    k_folds: int = 5,
    test_frac: float = 0.20,
    seed: int = 0,
    fit_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
    """K-fold subject CV. For each fold:
      - random 80/20 subject split per species (seeded)
      - stream train mean FC; derive test mean by subtraction from total
      - re-build C cost matrices on the train FC
      - fit a fresh model with full anchor supervision
      - FC-translation Pearson r on (train, test) FC

    Parameters
    ----------
    model_factory : callable returning a fresh FGWModel
    cache_dir : path to outputs/anndata/ (for load_cached + stream_mean_fc_subset)
    k_folds : int, default 5
    test_frac : float in (0, 1), default 0.20
    seed : int, default 0
    fit_kwargs : dict passed to model.fit()
    verbose : print per-fold progress

    Returns
    -------
    {
      'folds': [
        {'fold': 0, 'n_train_m', 'n_train_h', 'n_test_m', 'n_test_h',
         'train_r_overall', 'test_r_overall', 'gap', ...},
        ...
      ],
      'mean_train_r', 'mean_test_r', 'mean_gap', 'std_train_r', 'std_test_r',
    }
    """
    fit_kwargs = fit_kwargs or {}
    H, _ = load_cached("human", cache_dir=cache_dir)
    M, _ = load_cached("mouse", cache_dir=cache_dir)
    n_subj_m = M.uns["n_subjects"]; n_subj_h = H.uns["n_subjects"]
    fc_total_m = M.uns["fc_mean"].astype(np.float64)
    fc_total_h = H.uns["fc_mean"].astype(np.float64)
    n_obs_total_m = np.asarray(M.uns["fc_n_obs"]).astype(np.float64)  # (n_m, n_m)
    n_obs_total_h = np.asarray(H.uns["fc_n_obs"]).astype(np.float64)  # (n_h, n_h)

    idx_h = get_anchor_index(H.var)
    net_h = assign_networks(H.var, idx_h)

    folds = []
    for fold in range(k_folds):
        rng_m = np.random.default_rng(seed * 1000 + fold)
        rng_h = np.random.default_rng(seed * 1000 + fold + 500)
        test_m = rng_m.choice(n_subj_m,
                               size=max(1, int(round(test_frac * n_subj_m))),
                               replace=False)
        test_h = rng_h.choice(n_subj_h,
                               size=max(1, int(round(test_frac * n_subj_h))),
                               replace=False)
        train_m = np.setdiff1d(np.arange(n_subj_m), test_m)
        train_h = np.setdiff1d(np.arange(n_subj_h), test_h)

        # Build train means AND per-cell counts. Then derive test means by
        # count-aware subtraction:
        #   sum_test = sum_total - sum_train = total*n_total - train*n_train
        #   n_test   = n_total - n_train  (per-cell, since human n_obs is 100-113 not uniform)
        #   mean_test = sum_test / n_test
        # Old code assumed uniform coverage and divided by len(test_m), which is
        # slightly biased on the human side where some cells have <113 obs.
        fc_train_m, n_obs_train_m, _ = stream_mean_fc_subset("mouse", include_subjects=train_m)
        fc_train_h, n_obs_train_h, _ = stream_mean_fc_subset("human", include_subjects=train_h)
        n_obs_train_m = n_obs_train_m.astype(np.float64)
        n_obs_train_h = n_obs_train_h.astype(np.float64)

        sum_total_m = fc_total_m * n_obs_total_m
        sum_total_h = fc_total_h * n_obs_total_h
        sum_train_m = fc_train_m * n_obs_train_m
        sum_train_h = fc_train_h * n_obs_train_h

        n_obs_test_m = (n_obs_total_m - n_obs_train_m).clip(min=1)
        n_obs_test_h = (n_obs_total_h - n_obs_train_h).clip(min=1)

        fc_test_m = ((sum_total_m - sum_train_m) / n_obs_test_m).astype(np.float32)
        fc_test_h = ((sum_total_h - sum_train_h) / n_obs_test_h).astype(np.float32)

        # Patch the AnnDatas in-place (cheap, doesn't touch on-disk)
        M_train = M.copy(); H_train = H.copy()
        M_train.uns["fc_mean"] = fc_train_m
        H_train.uns["fc_mean"] = fc_train_h

        model = model_factory()
        model.fit(M_train, H_train, **fit_kwargs)

        train_metrics = fc_translation_quality(
            model.pi.astype(np.float64),
            fc_train_m.astype(np.float64),
            fc_train_h.astype(np.float64),
            network_labels_h=net_h,
        )
        test_metrics = fc_translation_quality(
            model.pi.astype(np.float64),
            fc_test_m.astype(np.float64),
            fc_test_h.astype(np.float64),
            network_labels_h=net_h,
        )
        folds.append({
            "fold": fold,
            "n_train_m": int(len(train_m)), "n_train_h": int(len(train_h)),
            "n_test_m":  int(len(test_m)),  "n_test_h":  int(len(test_h)),
            "train_r_overall": train_metrics["pearson_r_overall"],
            "test_r_overall":  test_metrics["pearson_r_overall"],
            "train_r_within_net": train_metrics.get("pearson_r_within_net", float("nan")),
            "test_r_within_net":  test_metrics.get("pearson_r_within_net", float("nan")),
            "gap": test_metrics["pearson_r_overall"] - train_metrics["pearson_r_overall"],
        })
        if verbose:
            f = folds[-1]
            print(f"  fold {fold}: train r={f['train_r_overall']:.3f} "
                  f"test r={f['test_r_overall']:.3f} gap={f['gap']:+.3f}")

    train_rs = np.array([f["train_r_overall"] for f in folds])
    test_rs  = np.array([f["test_r_overall"]  for f in folds])
    gaps     = np.array([f["gap"] for f in folds])
    return {
        "folds":         folds,
        "mean_train_r":  float(train_rs.mean()),
        "mean_test_r":   float(test_rs.mean()),
        "std_train_r":   float(train_rs.std()),
        "std_test_r":    float(test_rs.std()),
        "mean_gap":      float(gaps.mean()),
        "std_gap":       float(gaps.std()),
    }
