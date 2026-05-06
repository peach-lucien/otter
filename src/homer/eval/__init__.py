"""Evaluation: held-out anchor CV, subject-level CV, FC translation, nulls, bootstrap.

Public API at the package root for convenience:
    from homer.eval import fc_translation_quality, predict_human_fc
    from homer.eval import anchor_loo_cv, subject_kfold_cv
    from homer.eval import random_pi_null, permuted_anchor_null
    from homer.eval import bootstrap_pi
"""
from homer.eval.translation import (
    fc_translation_quality,
    predict_human_fc,
    random_pi_baseline,
    uniform_pi_baseline,
)
from homer.eval.anchor_cv import (
    anchor_loo_cv,
    held_out_metrics_graded,
)
from homer.eval.subject_cv import subject_kfold_cv
from homer.eval.nulls import (
    permuted_anchor_null,
    random_pi_null,
)
from homer.eval.bootstrap import bootstrap_pi
from homer.eval.full_space_metrics import (
    full_space_metrics,
    full_space_metrics_per_anchor,
)
from homer.eval.paired_tests import (
    compare_configs,
    mcnemar_paired_anchors,
    paired_bootstrap_diff,
)

__all__ = [
    "fc_translation_quality",
    "predict_human_fc",
    "random_pi_baseline",
    "uniform_pi_baseline",
    "anchor_loo_cv",
    "held_out_metrics_graded",
    "subject_kfold_cv",
    "permuted_anchor_null",
    "random_pi_null",
    "bootstrap_pi",
    "full_space_metrics",
    "full_space_metrics_per_anchor",
    "compare_configs",
    "mcnemar_paired_anchors",
    "paired_bootstrap_diff",
]
