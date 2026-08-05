"""Paired statistical comparisons between two configurations.

When two configs both report e.g. 81% top-1 across 42 anchors, "Config A is
better than Config B" is unjustified without a paired test. The differences
are typically a single anchor (~Bernoulli SE 0.06 for n=42), well within
random variation.

Public:
    mcnemar_paired_anchors(correct_a, correct_b)
        exact-binomial McNemar on the discordant pairs (anchors A got right
          but B got wrong, vs vice versa).

    paired_bootstrap_diff(correct_a, correct_b, *, n_boot=10000, seed=0)
        bootstrap CI on the difference in means + p-value via the
          permutation-style test on resampled paired differences.

    compare_configs(per_anchor_a, per_anchor_b)
        convenience: takes two boolean arrays of per-anchor correctness
          and returns a dict with both tests + summary.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def mcnemar_paired_anchors(correct_a, correct_b) -> dict:
    """Exact-binomial McNemar test on per-anchor binary correctness.

    Counts:
        b = anchors A got right, B got wrong
        c = anchors A got wrong, B got right

    Under H0 (no difference), b ~ Binomial(b+c, 0.5). Two-sided p-value is
    the exact binomial tail sum.

    Returns dict:
        n_concordant_correct, n_concordant_wrong, both got it right / both wrong
        n_a_only_correct, n_b_only_correct, discordant counts (b, c)
        n_discordant, b + c
        chi2. McNemar χ² statistic
        p_value, exact binomial p-value (two-sided)
        better, "A" / "B" / "neither" (effect direction)
    """
    a = np.asarray(correct_a).astype(bool)
    b_arr = np.asarray(correct_b).astype(bool)
    if a.shape != b_arr.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b_arr.shape}")

    both_right = int((a & b_arr).sum())
    both_wrong = int((~a & ~b_arr).sum())
    a_only     = int((a & ~b_arr).sum())   # b in McNemar notation
    b_only     = int((~a & b_arr).sum())   # c in McNemar notation
    n_disc     = a_only + b_only

    # χ² statistic with continuity correction
    if n_disc == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = (abs(a_only - b_only) - 1) ** 2 / n_disc if n_disc > 0 else 0.0
        # Exact binomial two-sided p-value
        from math import comb
        # P(X >= max(a_only, b_only) | n=n_disc, p=0.5) * 2
        k = max(a_only, b_only)
        p_one_tail = sum(comb(n_disc, i) for i in range(k, n_disc + 1)) / 2 ** n_disc
        p = min(1.0, 2 * p_one_tail)

    if a_only > b_only:
        better = "A"
    elif b_only > a_only:
        better = "B"
    else:
        better = "neither"
    return {
        "n":                       int(len(a)),
        "n_concordant_correct":    both_right,
        "n_concordant_wrong":      both_wrong,
        "n_a_only_correct":        a_only,
        "n_b_only_correct":        b_only,
        "n_discordant":            n_disc,
        "chi2":                    float(chi2),
        "p_value":                 float(p),
        "better":                  better,
        "diff_in_proportion":      float((a_only - b_only) / len(a)),
    }


def paired_bootstrap_diff(correct_a, correct_b, *,
                           n_boot: int = 10000, seed: int = 0,
                           ci_level: float = 0.95) -> dict:
    """Paired bootstrap CI on the difference (P(A correct) − P(B correct)).

    Resamples the paired anchors (with replacement) ``n_boot`` times and
    reports the percentile CI on the per-bootstrap difference in means.

    Returns dict with:
        observed_diff, ci_low, ci_high, p_value (two-sided), n_boot
    """
    a = np.asarray(correct_a).astype(int)
    b = np.asarray(correct_b).astype(int)
    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    observed = float((a - b).mean())
    alpha = 1 - ci_level
    ci_low  = float(np.quantile(diffs, alpha / 2))
    ci_high = float(np.quantile(diffs, 1 - alpha / 2))
    # Two-sided p via percentile of zero in the bootstrap distribution
    if observed >= 0:
        p_one = float((diffs <= 0).mean())
    else:
        p_one = float((diffs >= 0).mean())
    p_value = min(1.0, 2 * p_one)
    return {
        "observed_diff":  observed,
        "ci_low":         ci_low,
        "ci_high":        ci_high,
        "p_value":        p_value,
        "n_boot":         int(n_boot),
        "ci_level":       ci_level,
    }


def compare_configs(per_anchor_correct_a, per_anchor_correct_b,
                     *, name_a: str = "A", name_b: str = "B",
                     n_boot: int = 10000) -> dict:
    """Convenience: run McNemar + paired bootstrap on two per-anchor boolean arrays."""
    mc = mcnemar_paired_anchors(per_anchor_correct_a, per_anchor_correct_b)
    bp = paired_bootstrap_diff(per_anchor_correct_a, per_anchor_correct_b, n_boot=n_boot)
    a = np.asarray(per_anchor_correct_a).astype(bool)
    b = np.asarray(per_anchor_correct_b).astype(bool)
    return {
        "name_a":          name_a,
        "name_b":          name_b,
        "p_a":             float(a.mean()),
        "p_b":             float(b.mean()),
        "diff":            float(a.mean() - b.mean()),
        "mcnemar":         mc,
        "paired_bootstrap": bp,
        "verdict":         _verdict(mc, bp),
    }


def _verdict(mc: dict, bp: dict, alpha: float = 0.05) -> str:
    """Plain-English summary."""
    if mc["n_discordant"] == 0:
        return "identical performance, every anchor classified the same way"
    if mc["p_value"] < alpha and bp["p_value"] < alpha:
        return f"{mc['better']} significantly better (p < {alpha})"
    elif min(mc["p_value"], bp["p_value"]) < alpha:
        return f"weak signal, only one test < {alpha}; treat as tied"
    else:
        return f"no significant difference (both p ≥ {alpha}), treat as tied"
