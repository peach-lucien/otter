"""Bootstrap stability for FGW couplings.

Refits the model n_iter times on subject-level bootstrap resamples; aggregates
per-cell standard deviation of π. Stability = 1 - normalised_std (a per-cell
score in [0, 1]).

Public API:
    bootstrap_pi(model_factory, *, n_iter=40, seed=0) -> {pi_mean, pi_std, stability}
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from homer.data.io import load_cached, stream_mean_fc_subset


def bootstrap_pi(
    model_factory: Callable,
    cache_dir,
    *,
    n_iter: int = 40,
    seed: int = 0,
    fit_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
    """Bootstrap-resample subjects per species, refit model on each resample.

    Returns
    -------
    {
      'n_iter': int,
      'pi_mean': (n_m, n_h) float32,
      'pi_std':  (n_m, n_h) float32,
      'stability': (n_m, n_h) float32 in [0,1] (1 - std/std_max),
      'mean_stability': float,
      'frac_stable_above_0.8': float,
    }
    """
    fit_kwargs = fit_kwargs or {}
    H, _ = load_cached("human", cache_dir=cache_dir)
    M, _ = load_cached("mouse", cache_dir=cache_dir)
    n_subj_m = M.uns["n_subjects"]; n_subj_h = H.uns["n_subjects"]

    pi_sum  = None
    pi_sumsq = None
    rng = np.random.default_rng(seed)
    for it in range(n_iter):
        boot_m = rng.choice(n_subj_m, size=n_subj_m, replace=True)
        boot_h = rng.choice(n_subj_h, size=n_subj_h, replace=True)
        fc_m, _, _ = stream_mean_fc_subset("mouse", include_subjects=boot_m)
        fc_h, _, _ = stream_mean_fc_subset("human", include_subjects=boot_h)
        M_boot = M.copy(); H_boot = H.copy()
        M_boot.uns["fc_mean"] = fc_m; H_boot.uns["fc_mean"] = fc_h

        model = model_factory()
        model.fit(M_boot, H_boot, **fit_kwargs)
        pi = model.pi.astype(np.float64)
        if pi_sum is None:
            pi_sum  = pi.copy()
            pi_sumsq = pi * pi
        else:
            pi_sum  += pi
            pi_sumsq += pi * pi
        if verbose:
            print(f"  bootstrap iter {it+1}/{n_iter} done")

    n = float(n_iter)
    pi_mean = (pi_sum / n).astype(np.float32)
    pi_var  = (pi_sumsq / n) - (pi_mean.astype(np.float64) ** 2)
    pi_std  = np.sqrt(np.clip(pi_var, 0.0, None)).astype(np.float32)
    s_max = max(pi_std.max(), 1e-9)
    stability = (1.0 - pi_std / s_max).astype(np.float32)
    return {
        "n_iter": int(n_iter),
        "pi_mean": pi_mean,
        "pi_std":  pi_std,
        "stability": stability,
        "mean_stability": float(stability.mean()),
        "median_stability": float(np.median(stability)),
        "frac_stable_above_0.8": float((stability > 0.8).mean()),
        "frac_stable_above_0.5": float((stability > 0.5).mean()),
    }
