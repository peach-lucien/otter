"""Evaluation: held-out anchor CV, subject-level CV, FC translation, nulls, bootstrap.

Public API at the package root for convenience:
    from otter.eval import fc_translation_quality, predict_human_fc
    from otter.eval import anchor_loo_cv, subject_kfold_cv
    from otter.eval import random_pi_null, permuted_anchor_null
    from otter.eval import bootstrap_pi
"""
from otter.eval.translation import (
    fc_translation_quality,
    predict_human_fc,
    random_pi_baseline,
    uniform_pi_baseline,
)
from otter.eval.anchor_cv import (
    anchor_loo_cv,
    held_out_metrics_graded,
)
from otter.eval.subject_cv import subject_kfold_cv
from otter.eval.nulls import (
    permuted_anchor_null,
    random_pi_null,
)
from otter.eval.bootstrap import bootstrap_pi
from otter.eval.full_space_metrics import (
    full_space_metrics,
    full_space_metrics_per_anchor,
)
from otter.eval.paired_tests import (
    compare_configs,
    mcnemar_paired_anchors,
    paired_bootstrap_diff,
)
from otter.eval.region_level import (
    aggregate_pi_over_mouse_region,
    score_candidate_human_regions,
    fold_enrichment_candidate_regions,
    rank_candidate_regions,
    region_topk,
    evaluate_region_level,
    column_permuted_null,
    source_permuted_null,
    RegionLevelPairResult,
)
from otter.eval.trust_score import (
    compute_trust_score,
    compute_multisource_trust,
)
from otter.eval.network_coherence import (
    network_compactness,
    compare_network_compactness,
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
    "aggregate_pi_over_mouse_region",
    "score_candidate_human_regions",
    "fold_enrichment_candidate_regions",
    "rank_candidate_regions",
    "region_topk",
    "evaluate_region_level",
    "column_permuted_null",
    "source_permuted_null",
    "RegionLevelPairResult",
    "compute_trust_score",
    "compute_multisource_trust",
    "network_compactness",
    "compare_network_compactness",
]
